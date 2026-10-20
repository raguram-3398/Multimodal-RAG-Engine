"""Retrieve: question -> (rewrite / HyDE) -> BGE vector -> Pinecone -> list[Hit],
or with RETRIEVAL=hybrid: Pinecone and BM25 lists fused by RRF.
With RERANK=1 a cross-encoder re-scores those hits against the question.

Hit text comes from `chunks.jsonl`, the same file the live index was built from
(README, re-index step 4), not from Pinecone metadata. BM25 is built from the
same file at startup.
"""

import json
import os
import re
from collections.abc import Sequence
from pathlib import Path

import anthropic
from pinecone import Pinecone
from pinecone.errors import (
    PineconeConnectionError,
    PineconeTimeoutError,
    RateLimitError,
    ServiceError,
)
from rank_bm25 import BM25Okapi

from multimodal_rag.config import (
    BM25_EPSILON,
    CHUNKS_PATH,
    CLAUDE_MODEL,
    CLAUDE_TIMEOUT_S,
    EMBED_MODEL,
    HYDE,
    HYDE_MAX_TOKENS,
    INDEX_NAME,
    NAMESPACE,
    PINECONE_TIMEOUT_S,
    PRICE_IN_PER_MTOK,
    PRICE_OUT_PER_MTOK,
    QUERY_PREFIX,
    RERANK,
    RERANK_MODEL,
    RETRIEVAL,
    REWRITE,
    REWRITE_MAX_TOKENS,
    RRF_K,
    TOP_K,
)
from multimodal_rag.contracts import Chunk, Hit, SearchQuery


class QueryError(Exception):
    """The query path failed: Pinecone unreachable, index and chunk file out of
    sync, or Claude stopped early or errored. The message says which."""


def cost_usd(usage) -> float:
    """Dollar cost of one Haiku call from its token usage."""
    return (
        usage.input_tokens * PRICE_IN_PER_MTOK
        + usage.output_tokens * PRICE_OUT_PER_MTOK
    ) / 1e6


REWRITE_PROMPT = (
    "You turn a user's question into one search query for a semantic search over "
    "FDA presentations and books.\n"
    "Rules:\n"
    "- Keep the question's words, names, acronyms, numbers and years exactly as "
    "written.\n"
    "- If earlier questions from the same conversation are given, replace each "
    "pronoun or vague reference in the question (it, they, them, those, that "
    "letter, the center) with the words it refers to, copied from the earlier "
    "questions.\n"
    "- Add nothing else: no facts, laws, section numbers, years, numbers, acronym "
    "expansions, synonyms or extra keywords, even if you know them. Never answer "
    "the question.\n"
    "- If the question already makes sense on its own, return it unchanged.\n"
    "Record the query with the record_query tool."
)
# Forced tool call: the query arrives as a JSON field, so commentary cannot leak
# into it (in testing, a free-text reply once carried the model's reasoning), and a
# forced tool_choice turns thinking off.
REWRITE_TOOL = {
    "name": "record_query",
    "description": "Record the standalone search query.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}
HYDE_PROMPT = (
    "Write a passage of about 100 words, in the style of an FDA presentation slide "
    "or report, that answers the question below. Write it as document text, not "
    "as a reply to a person. Reply with the passage only."
)
GAP_PROMPT = (
    "A search over FDA presentations and books did not find everything needed to "
    "answer a question. Write one new search query that targets the missing "
    "information.\n"
    "Rules:\n"
    "- Keep the question's names, acronyms, numbers and years exactly as written.\n"
    "- Use the words of the missing information as described; add no facts or "
    "numbers of your own and never answer the question.\n"
    "Record the query with the record_query tool."
)


def _generate(
    system: str, user: str, max_tokens: int, client, tool: dict | None = None
) -> tuple[str, float]:
    """One Haiku call; (reply, cost_usd). With `tool`, the call is forced to use
    it and the reply is its `query` field; otherwise the reply is the text.
    Raises QueryError on an API error or an unexpected stop (`max_tokens`,
    `refusal`)."""
    forced = (
        {"tools": [tool], "tool_choice": {"type": "tool", "name": tool["name"]}}
        if tool
        else {}
    )
    try:
        msg = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            timeout=CLAUDE_TIMEOUT_S,
            **forced,
        )
    except anthropic.APIError as e:
        raise QueryError(f"Claude call failed: {type(e).__name__}: {e}") from e
    cost = cost_usd(msg.usage)
    if msg.stop_reason != ("tool_use" if tool else "end_turn"):
        raise QueryError(f"Claude stopped with {msg.stop_reason} (cost ${cost:.6f})")
    if tool:
        return next(b.input for b in msg.content if b.type == "tool_use")["query"], cost
    return "".join(b.text for b in msg.content if b.type == "text").strip(), cost


def rewrite(
    question: str, client: anthropic.Anthropic, history: Sequence[str] = ()
) -> tuple[str, float]:
    """Standalone search query for `question`, keeping its terms verbatim;
    (query, cost_usd). `history` = the conversation's earlier user questions,
    oldest first (answers are not passed)."""
    earlier = "".join(f"<earlier_question>{h}</earlier_question>\n" for h in history)
    return _generate(
        REWRITE_PROMPT,
        f"{earlier}<question>{question}</question>",
        REWRITE_MAX_TOKENS,
        client,
        REWRITE_TOOL,
    )


def gap_query(
    question: str, gap: str, client: anthropic.Anthropic
) -> tuple[str, float]:
    """Search query for what the self-check said the answer is missing;
    (query, cost_usd). Only searched: Claude still answers `question`."""
    return _generate(
        GAP_PROMPT,
        f"<question>{question}</question>\n<missing>{gap}</missing>",
        REWRITE_MAX_TOKENS,
        client,
        REWRITE_TOOL,
    )


def hypothetical_doc(question: str, client: anthropic.Anthropic) -> tuple[str, float]:
    """HyDE: one short invented FDA-style passage answering `question`;
    (passage, cost_usd). Its facts may be wrong; only its embedding is used."""
    return _generate(
        HYDE_PROMPT, f"<question>{question}</question>", HYDE_MAX_TOKENS, client
    )


def search_query(
    question: str,
    client: anthropic.Anthropic,
    history: Sequence[str] = (),
    rewrite_on: bool = REWRITE,
    hyde_on: bool = HYDE,
) -> SearchQuery:
    """What to embed for `question`.

    A follow-up (non-empty `history`) is always rewritten to a standalone
    question, which Claude then answers. Otherwise the rewrite (REWRITE) only
    changes what is searched. HyDE (HYDE) embeds a generated passage for the
    question, after any rewrite, on the passage side. Neither: the question as is,
    no Claude call.
    """
    text, cost = question, 0.0
    if history or rewrite_on:
        text, cost = rewrite(question, client, history)
    asked = text if history else question
    if hyde_on:
        passage, hyde_cost = hypothetical_doc(text, client)
        return SearchQuery(asked, passage, True, cost + hyde_cost)
    return SearchQuery(asked, text, False, cost)


def load_chunks(path: Path = CHUNKS_PATH) -> dict[str, Chunk]:
    """chunk_id -> Chunk for every line of `chunks.jsonl`."""
    with path.open(encoding="utf-8") as f:
        chunks = (Chunk(**json.loads(line)) for line in f)
        return {c.chunk_id: c for c in chunks}


def tokenize(text: str) -> list[str]:
    """BM25 tokens: lowercase alphanumeric runs. Used for chunks and queries alike.
    No stemming, no stopwords ("COVID-19" -> ["covid", "19"])."""
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: each id scores the sum of 1 / (k + rank) over the
    lists it appears in (rank 1-based). Best first; ties broken by id."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class Retriever:
    """Holds the BGE model, the Pinecone index handle, the chunk map, for hybrid
    the BM25 index over the same chunks and, with rerank, the cross-encoder.

    Build once per process; `search` is then one local embed + one Pinecone call
    (+ one local BM25 scoring pass for hybrid, + one local cross-encoder batch
    for rerank).
    """

    def __init__(self, strategy: str = RETRIEVAL, rerank: bool = RERANK) -> None:
        from sentence_transformers import CrossEncoder, SentenceTransformer

        if strategy not in ("vector", "hybrid"):
            raise ValueError(f"RETRIEVAL must be vector or hybrid, got {strategy!r}")
        self.strategy = strategy
        self.model = SentenceTransformer(EMBED_MODEL, device="cpu")
        self.chunks = load_chunks()
        if strategy == "hybrid":
            self.build_bm25()
        self.reranker = CrossEncoder(RERANK_MODEL, device="cpu") if rerank else None
        pc = Pinecone(
            api_key=os.environ["PINECONE_API_KEY"], timeout=PINECONE_TIMEOUT_S
        )
        self.index = pc.Index(INDEX_NAME)

    def build_bm25(self) -> None:
        """BM25Okapi over every chunk (text and vision): k1=1.5, b=0.75 (library
        defaults), terms in more than half the chunks score 0 (BM25_EPSILON)."""
        self.bm25_ids = list(self.chunks)
        self.bm25 = BM25Okapi(
            [tokenize(self.chunks[i].text) for i in self.bm25_ids],
            epsilon=BM25_EPSILON,
        )

    def embed(self, text: str, passage: bool = False) -> list[float]:
        """Normalized BGE vector (local, no network). Queries get QUERY_PREFIX;
        passages (the indexed chunks, a HyDE passage) get none."""
        prefix = "" if passage else QUERY_PREFIX
        return self.model.encode(prefix + text, normalize_embeddings=True).tolist()

    def search(
        self,
        text: str,
        k: int = TOP_K,
        passage: bool = False,
        question: str | None = None,
    ) -> list[Hit]:
        """Top-k chunks for `text`, best first, by the Retriever's strategy.

        vector: `vector_search` (score = cosine). hybrid: the top-k of
        `vector_search` and of `bm25_search` fused by `rrf` (score = RRF score,
        ~0.016-0.033, not comparable with cosine). With a reranker, those k hits
        are then reordered by `rerank` against `question` (default: `text`;
        pass the question when `text` is a HyDE passage). Raises as
        `vector_search`.
        """
        hits = self.vector_search(text, k, passage)
        if self.strategy == "hybrid":
            bm25 = self.bm25_search(text, k)
            fused = rrf(
                [[h.chunk.chunk_id for h in hits], [h.chunk.chunk_id for h in bm25]]
            )
            hits = [Hit(self.chunks[cid], score) for cid, score in fused[:k]]
        if self.reranker is None:
            return hits
        return self.rerank(text if question is None else question, hits)

    def rerank(self, question: str, hits: list[Hit]) -> list[Hit]:
        """`hits` reordered by the cross-encoder's score for (question, chunk
        text), best first; `Hit.score` becomes that score (a logit, -11.5 to
        10.8 on the 364 queries; higher = more relevant). Pairs over 512 tokens
        lose the end of the chunk. One local batch, no network; [] for no hits."""
        if not hits:
            return []
        scores = self.reranker.predict([(question, h.chunk.text) for h in hits])
        order = sorted(range(len(hits)), key=lambda i: -scores[i])
        return [Hit(hits[i].chunk, float(scores[i])) for i in order]

    def bm25_search(self, text: str, k: int = TOP_K) -> list[Hit]:
        """Top-k chunks by BM25 score (> 0 only), best first. A query sharing no
        token with any chunk returns [] rather than k arbitrary zero-score chunks.
        Local, no network."""
        scores = self.bm25.get_scores(tokenize(text))
        top = sorted(
            (i for i in range(len(scores)) if scores[i] > 0),
            key=lambda i: (-scores[i], self.bm25_ids[i]),
        )[:k]
        return [Hit(self.chunks[self.bm25_ids[i]], float(scores[i])) for i in top]

    def vector_search(
        self, text: str, k: int = TOP_K, passage: bool = False
    ) -> list[Hit]:
        """Top-k chunks by cosine similarity to `text`, best first.

        Raises QueryError if Pinecone times out, is unreachable, returns 5xx or
        429 (the SDK has already retried), or returns an id missing from
        chunks.jsonl.
        """
        try:
            res = self.index.query(
                vector=self.embed(text, passage),
                top_k=k,
                namespace=NAMESPACE,
                timeout=PINECONE_TIMEOUT_S,
            )
        except (
            PineconeTimeoutError,
            PineconeConnectionError,
            ServiceError,
            RateLimitError,
        ) as e:
            raise QueryError(f"Pinecone unavailable: {type(e).__name__}: {e}") from e
        hits = []
        for m in res.matches:
            if m.id not in self.chunks:
                raise QueryError(f"{m.id} is in {INDEX_NAME} but not in chunks.jsonl")
            hits.append(Hit(self.chunks[m.id], float(m.score)))
        return hits
