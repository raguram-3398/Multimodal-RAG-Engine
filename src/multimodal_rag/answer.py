"""Answer: hits -> abstain, or Claude Haiku answer with verified citations.

Try one question:  python -m multimodal_rag.answer "your question"
"""

import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import replace

import anthropic
from dotenv import load_dotenv

from multimodal_rag.config import (
    ABSTAIN_TEXT,
    ANSWER_K,
    ANSWER_MAX_TOKENS,
    CHECK_MAX_TOKENS,
    CLAUDE_MODEL,
    CLAUDE_TIMEOUT_S,
    PROJECT_DIR,
    REFUSAL_TEXT,
    SELF_CHECK,
    SELF_CHECK_MIN,
)
from multimodal_rag.contracts import Answer, Chunk, Citation, Hit
from multimodal_rag.retrieve import (
    QueryError,
    Retriever,
    cost_usd,
    gap_query,
    search_query,
)

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You answer questions about FDA documents using only the documents provided. "
    "Cite the passages that support each claim. Do not use outside knowledge. "
    "If the question asks about a specific version, edition, year or period and "
    "the documents only cover a different one, the documents do not contain the "
    "answer. "
    "If the documents do not contain the answer, reply with exactly this sentence "
    f"and nothing else: {ABSTAIN_TEXT}"
)
CHECK_PROMPT = (
    "You check an answer to a question about FDA documents. Using only the "
    "documents shown, score from 1 to 10 how completely and correctly the answer "
    "answers the question: 10 = fully answers it and every claim is supported by "
    "the documents; 1 = does not answer it or contradicts the documents. "
    "If the score is below 10, describe in one sentence the information that is "
    "missing or unsupported; otherwise leave the gap empty. "
    "Record your verdict with the record_check tool."
)
# Forced tool call: the score arrives as JSON (no parsing of prose) and a forced
# tool_choice turns thinking off, as for the rewriter and the judge.
CHECK_TOOL = {
    "name": "record_check",
    "description": "Record the 1-10 score and the missing information.",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {"type": "integer", "enum": list(range(1, 11))},
            "gap": {"type": "string"},
        },
        "required": ["score", "gap"],
    },
}


def doc_title(chunk: Chunk) -> str:
    """Document title Claude sees: the 1-indexed PDF page a reader would look up
    (data structures keep the 0-indexed page), marked when the text is a
    generated description rather than the PDF's own words."""
    page = f"{chunk.doc_id} p.{chunk.page + 1}"
    return f"generated visual description, {page}" if chunk.kind == "vision" else page


def build_documents(hits: list[Hit]) -> list[dict]:
    """One citable plain-text document per chunk; list position = document_index."""
    return [
        {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": h.chunk.text,
            },
            "title": doc_title(h.chunk),
            "citations": {"enabled": True},
        }
        for h in hits
    ]


def verify_citations(content: list, hits: list[Hit]) -> list[Citation]:
    """Citations from Claude's text blocks whose quote matches the chunk exactly.

    A citation is kept only if `cited_text == chunk.text[start:end]` for the chunk
    at its `document_index`; any other is dropped and logged. `generated` is set
    for quotes from vision chunks.
    """
    out: list[Citation] = []
    for block in content:
        for c in getattr(block, "citations", None) or []:
            chunk = hits[c.document_index].chunk
            span = chunk.text[c.start_char_index : c.end_char_index]
            if c.type != "char_location" or span != c.cited_text:
                log.warning("dropped unverified citation of %s", chunk.chunk_id)
                continue
            score = hits[c.document_index].score
            generated = chunk.kind == "vision"
            out.append(
                Citation(
                    chunk.chunk_id, chunk.doc_id, chunk.page, span, score, generated
                )
            )
    return out


def answer(question: str, hits: list[Hit], client: anthropic.Anthropic) -> Answer:
    """Answer from the top ANSWER_K hits with verified citations; abstain without
    calling Claude only if there are no hits. Claude decides every other abstain
    (no score threshold since Day 30: it caught no unanswerable query Claude
    missed and blocked answerable ones with the gold page in the top 10).

    If Claude stops with `refusal`, returns REFUSAL_TEXT with `refused=True`:
    partial text and citations are discarded, the cost is kept. Raises
    QueryError if the call fails or stops any other way than `end_turn`
    (e.g. `max_tokens` cut the answer and its citations short).
    """
    t0 = time.monotonic()
    if not hits:
        return Answer(ABSTAIN_TEXT, [], True, False, 0.0, time.monotonic() - t0)

    hits = hits[:ANSWER_K]
    try:
        msg = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=ANSWER_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": build_documents(hits)
                    + [{"type": "text", "text": question}],
                }
            ],
            timeout=CLAUDE_TIMEOUT_S,
        )
    except anthropic.APIError as e:
        raise QueryError(f"Claude call failed: {type(e).__name__}: {e}") from e

    cost = cost_usd(msg.usage)
    log.info(
        "claude %s in=%d out=%d cost=$%.6f",
        msg.stop_reason,
        msg.usage.input_tokens,
        msg.usage.output_tokens,
        cost,
    )
    if msg.stop_reason == "refusal":
        return Answer(REFUSAL_TEXT, [], False, True, cost, time.monotonic() - t0)
    if msg.stop_reason != "end_turn":
        raise QueryError(f"Claude stopped with {msg.stop_reason} (cost ${cost:.6f})")

    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    citations = verify_citations(msg.content, hits)
    abstained = text == ABSTAIN_TEXT and not citations
    return Answer(text, citations, abstained, False, cost, time.monotonic() - t0)


def answered(a: Answer) -> bool:
    """Claude answered from the documents: neither an abstain nor a refusal."""
    return not (a.abstained or a.refused)


def self_check(
    question: str, ans: Answer, hits: list[Hit], client: anthropic.Anthropic
) -> tuple[int, str, float]:
    """Haiku scores `ans` 1-10 against the question and the ANSWER_K chunks it
    was written from; (score, gap, cost_usd). `gap` = one sentence on what is
    missing or unsupported ("" at 10). Raises QueryError on an API error or a
    score outside 1-10."""
    docs = "\n".join(
        f'<document title="{doc_title(h.chunk)}">{h.chunk.text}</document>'
        for h in hits[:ANSWER_K]
    )
    prompt = (
        f"<documents>\n{docs}\n</documents>\n"
        f"<question>{question}</question>\n<answer>{ans.text}</answer>"
    )
    try:
        msg = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CHECK_MAX_TOKENS,
            system=CHECK_PROMPT,
            tools=[CHECK_TOOL],
            tool_choice={"type": "tool", "name": CHECK_TOOL["name"]},
            messages=[{"role": "user", "content": prompt}],
            timeout=CLAUDE_TIMEOUT_S,
        )
    except anthropic.APIError as e:
        raise QueryError(f"self-check failed: {type(e).__name__}: {e}") from e
    cost = cost_usd(msg.usage)
    if msg.stop_reason != "tool_use":
        raise QueryError(f"self-check stopped with {msg.stop_reason} (${cost:.6f})")
    verdict = next(b.input for b in msg.content if b.type == "tool_use")
    if verdict.get("score") not in range(1, 11):
        raise QueryError(f"self-check returned an invalid score: {verdict}")
    return verdict["score"], verdict.get("gap", ""), cost


def check_and_retry(
    question: str,
    first: Answer,
    hits: list[Hit],
    retriever: Retriever,
    client: anthropic.Anthropic,
) -> tuple[Answer, list[Hit]]:
    """Self-check `first`; below SELF_CHECK_MIN, search once more and answer again.

    An abstain or a refusal is final and not checked. Otherwise the score becomes
    `confidence`. Below SELF_CHECK_MIN: the gap becomes a new search query
    (`gap_query`), searched with the normal pipeline and reranked against
    `question`, and Claude answers `question` from the new hits. That second
    answer is final, even an abstain; it is checked once more only to fill
    `confidence` (`retried=True`). Returns the final answer, with the cost and
    latency of every step, and the ANSWER_K hits it was written from.
    """
    if not answered(first):
        return first, hits[:ANSWER_K]
    t0 = time.monotonic()
    score, gap, check_cost = self_check(question, first, hits, client)
    if score >= SELF_CHECK_MIN:
        return (
            replace(
                first,
                cost_usd=first.cost_usd + check_cost,
                latency_s=first.latency_s + time.monotonic() - t0,
                confidence=score,
            ),
            hits[:ANSWER_K],
        )
    query, query_cost = gap_query(question, gap, client)
    new_hits = retriever.search(query, question=question)
    second = answer(question, new_hits, client)
    cost = first.cost_usd + check_cost + query_cost + second.cost_usd
    confidence = None
    if answered(second):
        confidence, _, second_check_cost = self_check(
            question, second, new_hits, client
        )
        cost += second_check_cost
    log.info("self-check %d < %d, retried with %r", score, SELF_CHECK_MIN, query)
    final = replace(
        second,
        cost_usd=cost,
        latency_s=first.latency_s + time.monotonic() - t0,
        confidence=confidence,
        retried=True,
    )
    return final, new_hits[:ANSWER_K]


def ask(
    question: str,
    retriever: Retriever,
    client: anthropic.Anthropic,
    history: Sequence[str] = (),
    self_check_on: bool = SELF_CHECK,
) -> tuple[Answer, list[Hit]]:
    """Transform the question (follow-up rewrite, REWRITE, HYDE), retrieve, answer,
    and with SELF_CHECK check the answer and retry once (`check_and_retry`).

    `history` = the conversation's earlier user questions, oldest first.
    Returns the answer and the ANSWER_K hits it was written from (after a
    retry, the retry's). `cost_usd` includes the rewrite / HyDE calls;
    `latency_s` covers all steps.
    """
    t0 = time.monotonic()
    sq = search_query(question, client, history)
    hits = retriever.search(sq.text, passage=sq.passage, question=sq.question)
    result = answer(sq.question, hits, client)
    hits = hits[:ANSWER_K]
    if self_check_on:
        result, hits = check_and_retry(sq.question, result, hits, retriever, client)
    final = replace(
        result,
        cost_usd=result.cost_usd + sq.cost_usd,
        latency_s=time.monotonic() - t0,
    )
    return final, hits


def main() -> None:
    """Answer the question given on the command line and print its citations."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    log.setLevel(logging.INFO)  # this module's cost line; not the HTTP libraries'
    load_dotenv(PROJECT_DIR / ".env")
    result, _ = ask(sys.argv[1], Retriever(), anthropic.Anthropic())
    print(result.text, "\n")
    for c in result.citations:
        where = f"{c.doc_id} p.{c.page + 1} [{c.chunk_id}, {c.score:.3f}]"
        mark = " (generated description)" if c.generated else ""
        print(f"- {where}{mark} {c.cited_text!r}")
    print(
        f"\nabstained={result.abstained} refused={result.refused} "
        f"confidence={result.confidence} retried={result.retried} "
        f"cost=${result.cost_usd:.6f} "
        f"latency={result.latency_s:.2f}s"
    )


if __name__ == "__main__":
    main()
