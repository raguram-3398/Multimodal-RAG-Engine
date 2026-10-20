"""Evaluate the pipeline.

Free retrieval eval (no Claude): page-level recall@10 and NDCG@10 on all 364
English queries, split by visual content type, and the fixed eval sample.
    python -m multimodal_rag.evaluate

Both follow the REWRITE / HYDE flags, RETRIEVAL and RERANK,
e.g. `RERANK=1 python -m ...`.

Ten two-turn conversations (Claude rewrites the follow-ups; retrieval metrics only):
    python -m multimodal_rag.evaluate followups

Full eval (Claude): answers the fixed 60 + 20 sample, scores it with RAGAS, the
Haiku judge and the abstain rule, gates it against data/baseline.json (not
gated while that file is missing) and appends it to data/scores.jsonl. With
SELF_CHECK=1 each answer is self-checked and retried once below
SELF_CHECK_MIN; the retried queries' first answers are scored too, for a paired
before/after.
    python -m multimodal_rag.evaluate full "label"

Baseline: data/baseline.json from the last BASELINE_RUNS (5) ungated full runs of
unchanged code (refuses to overwrite an existing baseline):
    python -m multimodal_rag.evaluate baseline "reason"
"""

import hashlib
import json
import logging
import math
import platform
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from multimodal_rag.answer import answer, answered, check_and_retry
from multimodal_rag.config import (
    ANSWER_K,
    BASELINE_PATH,
    BASELINE_RUNS,
    CHUNKS_PATH,
    CLAUDE_MODEL,
    CLAUDE_TIMEOUT_S,
    CORPUS_INDEX_PATH,
    EMBED_MODEL,
    EVAL_DIR,
    EVAL_K,
    EVAL_WORKERS,
    FOLLOWUPS_PATH,
    GATE_FLOORS,
    HYDE,
    INDEX_NAME,
    JUDGE_MAX_TOKENS,
    PRICE_IN_PER_MTOK,
    PRICE_OUT_PER_MTOK,
    PROJECT_DIR,
    QRELS_PATH,
    QUERIES_PATH,
    RAGAS_MAX_TOKENS,
    RERANK,
    RERANK_MODEL,
    RETRIEVAL,
    REWRITE,
    SAMPLE_PATH,
    SAMPLE_SEED,
    SAMPLE_SIZE,
    SCORES_PATH,
    SELF_CHECK,
    SELF_CHECK_MIN,
    UNANSWERABLE_PATH,
    VISION_CONTENT,
    VISION_PATH,
    VISUAL_CONTENT,
)
from multimodal_rag.contracts import Answer, Hit, SearchQuery
from multimodal_rag.retrieve import QueryError, Retriever, cost_usd, search_query

Page = tuple[str, int]  # (doc_id, 0-indexed page)


def load_queries() -> pd.DataFrame:
    """The 364 English queries with `gold` = {(doc_id, page): grade 1 or 2} and
    `visual` = a relevant page holds chart, infographic, table or image content."""
    q = pd.read_parquet(QUERIES_PATH).query("language == 'english'").copy()
    q["query_id"] = q["query_id"].astype(int)
    corpus = pd.read_parquet(CORPUS_INDEX_PATH).set_index("corpus_id")
    qrels = pd.read_parquet(QRELS_PATH)
    qrels = qrels[qrels["query_id"].isin(q["query_id"])]

    gold: dict[int, dict[Page, int]] = defaultdict(dict)
    for r in qrels.itertuples():
        row = corpus.loc[r.corpus_id]
        gold[r.query_id][(row.doc_id, int(row.page_number_in_doc))] = int(r.score)
    q["gold"] = q["query_id"].map(gold)
    q["visual"] = q["content_type"].map(lambda ct: bool(VISUAL_CONTENT & set(ct)))
    return q.reset_index(drop=True)


def make_sample(queries: pd.DataFrame) -> dict:
    """The fixed eval sample: SAMPLE_SIZE answerable query ids + all unanswerable ids.

    Strata = query_type_for_generation x visual; each stratum gets its share of
    SAMPLE_SIZE by largest remainder. Written once to SAMPLE_PATH; if the file
    exists it is returned unchanged, so the sample never moves once written.
    """
    if SAMPLE_PATH.is_file():
        return json.loads(SAMPLE_PATH.read_text())

    strata = queries.groupby(["query_type_for_generation", "visual"])["query_id"]
    groups = {key: sorted(ids) for key, ids in strata}
    exact = {k: SAMPLE_SIZE * len(ids) / len(queries) for k, ids in groups.items()}
    alloc = {k: int(v) for k, v in exact.items()}
    by_remainder = sorted(exact, key=lambda k: exact[k] - alloc[k], reverse=True)
    for k in by_remainder[: SAMPLE_SIZE - sum(alloc.values())]:
        alloc[k] += 1

    rng = random.Random(SAMPLE_SEED)
    answerable = sorted(
        qid for k in sorted(groups) for qid in rng.sample(groups[k], alloc[k])
    )
    unanswerable = [
        json.loads(line)["id"] for line in UNANSWERABLE_PATH.open(encoding="utf-8")
    ]
    sample = {
        "seed": SAMPLE_SEED,
        "answerable": answerable,
        "unanswerable": unanswerable,
    }
    SAMPLE_PATH.write_text(json.dumps(sample, indent=1) + "\n")
    return sample


def ranked_pages(hits: list[Hit], k: int = EVAL_K) -> list[Page]:
    """First k distinct (doc_id, page) in hit order: a book page yields ~3 chunks,
    so scoring chunks directly would crowd out other pages."""
    pages: list[Page] = []
    for h in hits:
        p = (h.chunk.doc_id, h.chunk.page)
        if p not in pages:
            pages.append(p)
        if len(pages) == k:
            break
    return pages


def recall_at_k(ranked: list[Page], gold: dict[Page, int], k: int = EVAL_K) -> float:
    """Share of all gold pages found in the top k (queries with > k gold pages
    cannot reach 1.0)."""
    return len(set(ranked[:k]) & gold.keys()) / len(gold)


def ndcg_at_k(ranked: list[Page], gold: dict[Page, int], k: int = EVAL_K) -> float:
    """NDCG with gain = qrel grade (1 critically, 2 fully relevant)."""
    dcg = sum(gold.get(p, 0) / math.log2(i + 2) for i, p in enumerate(ranked[:k]))
    ideal = sorted(gold.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg


def recall_by_content(ret: pd.DataFrame) -> dict[str, float]:
    """recall@10 over the queries whose relevant pages hold each VISION_CONTENT
    type (a query can count for several). Logged, not gated."""
    return {
        f"recall@10 {t}": float(
            ret.recall[ret.content_type.map(lambda c: t in c)].mean()
        )
        for t in VISION_CONTENT
    }


def search_queries(
    questions: list[str], client: anthropic.Anthropic
) -> list[SearchQuery]:
    """`search_query` for each question under the REWRITE / HYDE flags, EVAL_WORKERS
    Claude calls at a time (no call when both flags are off)."""
    with ThreadPoolExecutor(EVAL_WORKERS) as pool:
        return list(pool.map(lambda q: search_query(q, client), questions))


def retrieval_scores(
    queries: pd.DataFrame, retriever: Retriever, client: anthropic.Anthropic
) -> pd.DataFrame:
    """Per query: page-level recall@10, NDCG@10, whether the top chunks held
    fewer than EVAL_K distinct pages, the query's content types and format, the
    rewrite / HyDE cost and the wall time of `retriever.search` in ms (embed,
    Pinecone, BM25, RRF, rerank; not the rewrite / HyDE calls)."""
    rows = []
    sqs = search_queries(list(queries["query"]), client)
    for q, sq in zip(queries.itertuples(), sqs):
        t0 = time.perf_counter()
        hits = retriever.search(sq.text, passage=sq.passage, question=sq.question)
        search_ms = (time.perf_counter() - t0) * 1000
        ranked = ranked_pages(hits)
        rows.append(
            {
                "query_id": q.query_id,
                "recall": recall_at_k(ranked, q.gold),
                "ndcg": ndcg_at_k(ranked, q.gold),
                "short": len(ranked) < EVAL_K,
                "content_type": set(q.content_type),
                "format": q.query_format,
                "cost": sq.cost_usd,
                "search_ms": search_ms,
            }
        )
    return pd.DataFrame(rows)


def conversation_scores(
    retriever: Retriever, client: anthropic.Anthropic
) -> pd.DataFrame:
    """recall@10 / NDCG@10 of each follow-up in FOLLOWUPS_PATH three ways:
    `raw` = the follow-up searched as is, `rewritten` = rewritten with the first
    question as history, `original` = the dataset query it was written from (the
    ceiling). Gold pages are the original query's."""
    queries = load_queries().set_index("query_id")
    convs = [json.loads(line) for line in FOLLOWUPS_PATH.open(encoding="utf-8")]

    def rewrite_one(c: dict) -> SearchQuery:
        return search_query(c["follow_up"], client, [c["first"]], False, False)

    with ThreadPoolExecutor(EVAL_WORKERS) as pool:
        sqs = list(pool.map(rewrite_one, convs))
    rows = []
    for c, sq in zip(convs, sqs):
        gold = queries.at[c["source_query_id"], "gold"]
        row = {"id": c["id"], "rewritten_query": sq.text, "cost": sq.cost_usd}
        texts = {
            "raw": c["follow_up"],
            "rewritten": sq.text,
            "original": queries.at[c["source_query_id"], "query"],
        }
        for way, text in texts.items():
            ranked = ranked_pages(retriever.search(text))
            row[f"recall {way}"] = recall_at_k(ranked, gold)
            row[f"ndcg {way}"] = ndcg_at_k(ranked, gold)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- full eval


@dataclass(frozen=True)
class Result:
    """One sample question run through the pipeline. `reference` is None for
    the unanswerable queries; `hits` are the ANSWER_K chunks Claude saw.
    `first` = the same question before its self-check retry, kept to
    score the retry pairwise; None when there was no retry."""

    query_id: str
    question: str
    reference: str | None
    hits: list[Hit]
    answer: Answer
    first: "Result | None" = None


def run_sample(
    queries: pd.DataFrame,
    sample: dict,
    retriever: Retriever,
    client: anthropic.Anthropic,
    self_check_on: bool = SELF_CHECK,
) -> list[Result]:
    """Answer the 60 answerable then the 20 unanswerable sample queries.

    Each question goes through `search_query` (REWRITE / HYDE flags); its cost is
    added to the answer's. Search runs in order (one local embed + one Pinecone
    call each); the Claude calls run EVAL_WORKERS at a time. With SELF_CHECK each
    answer then goes through `check_and_retry`, one query at a time (a retry
    searches, and the Retriever is not shared across threads). Raises QueryError
    like `answer`, naming the query.
    """
    by_id = queries.set_index("query_id")
    items = [
        (str(i), by_id.at[i, "query"], by_id.at[i, "answer"])
        for i in sample["answerable"]
    ]
    for line in UNANSWERABLE_PATH.open(encoding="utf-8"):
        u = json.loads(line)
        if u["id"] in sample["unanswerable"]:
            items.append((u["id"], u["query"], None))

    def answer_one(item: tuple, item_hits: list[Hit], sq: SearchQuery) -> Answer:
        try:
            a = answer(sq.question, item_hits, client)
        except QueryError as e:
            raise QueryError(f"query {item[0]}: {e}") from e
        return replace(a, cost_usd=a.cost_usd + sq.cost_usd)

    sqs = search_queries([q for _, q, _ in items], client)
    hits = [
        retriever.search(sq.text, passage=sq.passage, question=sq.question)
        for sq in sqs
    ]
    with ThreadPoolExecutor(EVAL_WORKERS) as pool:
        answers = list(pool.map(answer_one, items, hits, sqs))
    results = [
        Result(qid, q, ref, h[:ANSWER_K], a)
        for (qid, q, ref), h, a in zip(items, hits, answers)
    ]
    if not self_check_on:
        return results
    checked = []
    for r, sq, h in zip(results, sqs, hits):
        try:
            final, final_hits = check_and_retry(
                sq.question, r.answer, h, retriever, client
            )
        except QueryError as e:
            raise QueryError(f"query {r.query_id}: {e}") from e
        first = r if final.retried else None
        checked.append(replace(r, hits=final_hits, answer=final, first=first))
    return checked


def ragas_llm(api_key: str | None = None):
    """Haiku as the RAGAS judge, through LangChain (`bypass_temperature`: Haiku
    5.5 rejects the parameter). `api_key` None = ANTHROPIC_API_KEY from the
    environment (the eval); the app always passes the visitor's key."""
    from langchain_anthropic import ChatAnthropic
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(
        ChatAnthropic(
            model=CLAUDE_MODEL,
            max_tokens=RAGAS_MAX_TOKENS,
            timeout=CLAUDE_TIMEOUT_S,
            api_key=api_key,
        ),
        bypass_temperature=True,
    )


def ragas_embeddings():
    """BGE for answer relevancy (local, CPU). Loads the model: build once."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )


def ragas_run(
    rows: list[Result], metrics: list, llm, emb
) -> tuple[pd.DataFrame, float]:
    """Score `rows` with the RAGAS `metrics`; (per-row scores, dollar cost).
    The contexts are each row's hits; a score RAGAS could not compute is NaN."""
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.cost import get_token_usage_for_anthropic

    data = EvaluationDataset.from_list(
        [
            {
                "user_input": r.question,
                "response": r.answer.text,
                "retrieved_contexts": [h.chunk.text for h in r.hits],
                "reference": r.reference,
            }
            for r in rows
        ]
    )
    out = evaluate(
        data,
        metrics=metrics,
        llm=llm,
        embeddings=emb,
        run_config=RunConfig(max_workers=EVAL_WORKERS),
        token_usage_parser=get_token_usage_for_anthropic,
        show_progress=False,
    )
    cost = out.total_cost(
        cost_per_input_token=PRICE_IN_PER_MTOK / 1e6,
        cost_per_output_token=PRICE_OUT_PER_MTOK / 1e6,
    )
    return out.to_pandas(), float(cost)


def ragas_scores(results: list[Result]) -> tuple[pd.DataFrame, float]:
    """RAGAS scores per answerable result, plus the RAGAS dollar cost.

    Context precision and context recall (retrieved chunks vs the reference) are
    scored on every answerable result. Faithfulness and answer relevancy judge
    the answer text, so they are scored only where the pipeline answered: an
    abstain has no claims to check, yet RAGAS scored false abstains 0.65
    faithfulness on average (up to 1.0) early on. Abstains are NaN in those two
    columns by design and are counted by `abstain_scores` instead.

    Judge = `ragas_llm()`; embeddings = BGE, for answer relevancy only. A score
    RAGAS could not compute is also NaN; the caller reports those separately.
    """
    from ragas.metrics import (
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        answer_relevancy,
        faithfulness,
    )

    llm, emb = ragas_llm(), ragas_embeddings()
    answerable = [r for r in results if r.reference is not None]
    scored = [i for i, r in enumerate(answerable) if answered(r.answer)]
    ctx, ctx_cost = ragas_run(
        answerable, [LLMContextPrecisionWithReference(), LLMContextRecall()], llm, emb
    )
    ans, ans_cost = ragas_run(
        [answerable[i] for i in scored], [faithfulness, answer_relevancy], llm, emb
    )
    df = pd.DataFrame(
        {
            "faithfulness": np.nan,
            "answer_relevancy": np.nan,
            "context_precision": ctx["llm_context_precision_with_reference"],
            "context_recall": ctx["context_recall"],
        }
    )
    df.loc[scored, "faithfulness"] = ans["faithfulness"].to_numpy()
    df.loc[scored, "answer_relevancy"] = ans["answer_relevancy"].to_numpy()
    return df, ctx_cost + ans_cost


def live_scores(
    question: str, ans: Answer, hits: list[Hit], api_key: str, emb
) -> tuple[float, float, float]:
    """RAGAS faithfulness and answer relevancy of one answered question (the
    app), judged by Haiku on `api_key`; (faithfulness, answer_relevancy,
    cost_usd). No reference exists for a visitor's question, so the
    reference-based metrics are not computed. NaN = RAGAS could not score it.
    """
    from ragas.metrics import answer_relevancy, faithfulness

    row = Result("live", question, None, hits, ans)
    df, cost = ragas_run(
        [row], [faithfulness, answer_relevancy], ragas_llm(api_key), emb
    )
    return float(df.faithfulness[0]), float(df.answer_relevancy[0]), cost


JUDGE_CRITERIA = ("faithfulness", "relevance", "completeness")
JUDGE_PROMPT = """You grade an answer to a question about FDA documents against a \
reference answer written from the same documents. Score each criterion 1-5.

- faithfulness: are the answer's facts consistent with the reference? 5 = nothing \
contradicts it, 1 = its main claim contradicts it. Facts the reference does not \
mention are not errors.
- relevance: does it answer the question that was asked? 5 = directly, 1 = not at all.
- completeness: does it contain the reference's key facts? 5 = all, 1 = none.

Do not reward length. If the answer says the documents do not contain the answer, \
score 1 on all three.

<question>{question}</question>
<reference>{reference}</reference>
<answer>{answer}</answer>"""
JUDGE_TOOL = {
    "name": "record_scores",
    "description": "Record the 1-5 scores and a one-sentence reason.",
    "input_schema": {
        "type": "object",
        "properties": {
            **{c: {"type": "integer", "enum": [1, 2, 3, 4, 5]} for c in JUDGE_CRITERIA},
            "reason": {"type": "string"},
        },
        "required": [*JUDGE_CRITERIA, "reason"],
    },
}


def judge(
    question: str, reference: str, ans: Answer, client: anthropic.Anthropic
) -> tuple[dict, float]:
    """Haiku scores `ans` 1-5 on JUDGE_CRITERIA against the reference answer.

    Returns ({criterion: score, "reason": str}, cost_usd). The tool call is forced,
    so the scores arrive as JSON matching JUDGE_TOOL. Raises QueryError on an API
    error or a score outside 1-5.
    """
    prompt = JUDGE_PROMPT.format(
        question=question, reference=reference, answer=ans.text
    )
    try:
        msg = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=JUDGE_MAX_TOKENS,
            tools=[JUDGE_TOOL],
            tool_choice={"type": "tool", "name": JUDGE_TOOL["name"]},
            messages=[{"role": "user", "content": prompt}],
            timeout=CLAUDE_TIMEOUT_S,
        )
    except anthropic.APIError as e:
        raise QueryError(f"judge call failed: {type(e).__name__}: {e}") from e
    scores = next(b.input for b in msg.content if b.type == "tool_use")
    if any(scores.get(c) not in range(1, 6) for c in JUDGE_CRITERIA):
        raise QueryError(f"judge returned invalid scores: {scores}")
    return scores, cost_usd(msg.usage)


def abstain_scores(results: list[Result]) -> dict[str, float]:
    """Abstain accuracy both ways, from `Answer.abstained` (ABSTAIN_TEXT with no
    citations): share of unanswerable results that abstained,
    and share of answerable results that were answered. A refusal is neither: it
    counts against both."""
    ua = [r.answer.abstained for r in results if r.reference is None]
    ans = [answered(r.answer) for r in results if r.reference is not None]
    return {
        "abstain_unanswerable": sum(ua) / len(ua),
        "answer_answerable": sum(ans) / len(ans),
    }


def gate(current: dict[str, float], baseline: dict) -> list[str]:
    """Names of gated metrics (the baseline's `thresholds`) whose current value is
    below the baseline mean by more than the threshold. A NaN current value
    fails: it cannot be shown not to drop."""
    return [
        m
        for m, thr in baseline["thresholds"].items()
        if math.isnan(current[m])
        or baseline["metrics"][m]["mean"] - current[m] > thr + 1e-9
    ]


def retry_scores(
    results: list[Result], per_query: pd.DataFrame, client: anthropic.Anthropic
) -> tuple[dict, float]:
    """Self-check summary and the retried answerable queries scored
    pairwise: their first answer and hits (RAGAS + judge, scored here) vs their
    final ones (from `per_query`). Returns (summary, cost of the extra scoring).
    """
    retried = [r for r in results if r.answer.retried]
    pairs = [r for r in retried if r.reference is not None]
    conf = [r.answer.confidence for r in results if r.answer.confidence is not None]
    summary = {
        "min_score": SELF_CHECK_MIN,
        "n_checked": sum(
            answered(r.first.answer if r.first else r.answer) for r in results
        ),
        "n_retried": len(retried),
        "retried_ids": [r.query_id for r in retried],
        "n_retried_unanswerable": len(retried) - len(pairs),
        "answered_to_abstained": [r.query_id for r in retried if r.answer.abstained],
        "final_confidence_counts": {str(s): conf.count(s) for s in sorted(set(conf))},
    }
    if not pairs:
        return summary, 0.0
    firsts = [r.first for r in pairs]
    ragas_df, ragas_cost = ragas_scores(firsts)
    with ThreadPoolExecutor(EVAL_WORKERS) as pool:
        judged = list(
            pool.map(lambda r: judge(r.question, r.reference, r.answer, client), firsts)
        )
    before = pd.concat(
        [ragas_df, pd.DataFrame([j for j, _ in judged]).add_prefix("judge_")], axis=1
    )
    after = per_query.set_index("query_id").loc[[r.query_id for r in pairs]]
    cols = [*ragas_df.columns, *(f"judge_{c}" for c in JUDGE_CRITERIA)]
    summary["pairs"] = {
        c: {
            "first": float(before[c].mean()),
            "final": float(after[c].mean()),
        }
        for c in cols
    }
    summary["pairs_per_query"] = {
        r.query_id: {
            c: [float(before[c].iloc[i]), float(after.at[r.query_id, c])] for c in cols
        }
        for i, r in enumerate(pairs)
    }
    return summary, ragas_cost + sum(c for _, c in judged)


def fingerprint() -> str:
    """sha256 over the files a run's scores depend on: package code, uv.lock,
    chunks, vision descriptions, eval files, the dataset's queries and qrels.
    Equal fingerprints = same inputs (the Pinecone index is recorded by name)."""
    paths = [
        *sorted(Path(__file__).parent.glob("*.py")),
        PROJECT_DIR / "uv.lock",
        CHUNKS_PATH,
        VISION_PATH,
        *sorted(EVAL_DIR.glob("*.json*")),
        QUERIES_PATH,
        QRELS_PATH,
    ]
    h = hashlib.sha256()
    for p in paths:
        h.update(p.relative_to(PROJECT_DIR).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def run_env() -> dict[str, str]:
    """Models, RAGAS version and machine of a run, logged with its scores."""
    return {
        "claude_model": CLAUDE_MODEL,
        "embed_model": EMBED_MODEL,
        "rerank_model": RERANK_MODEL,
        "ragas": version("ragas"),
        "machine": platform.platform(),
    }


def load_baseline() -> dict | None:
    """data/baseline.json, or None if there is no baseline yet."""
    if not BASELINE_PATH.is_file():
        return None
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def make_baseline(runs: list[dict], reason: str) -> dict:
    """Baseline from BASELINE_RUNS ungated runs of the same inputs: mean, min,
    max and std (ddof 1) of every logged metric, and per gated metric threshold
    = max(floor, max - min). Raises ValueError on too few runs, a gated run, or
    runs that differ in index, flags, env or fingerprint."""
    if len(runs) < BASELINE_RUNS:
        raise ValueError(f"need {BASELINE_RUNS} runs, got {len(runs)}")
    if any(r["passed"] is not None for r in runs):
        raise ValueError("a gated run is among them: baseline runs are ungated")
    same = ("index", "flags", "env", "fingerprint")
    for k in same:
        if any(r.get(k) != runs[0].get(k) for r in runs):
            raise ValueError(f"runs differ in {k}")
    df = pd.DataFrame([r["metrics"] for r in runs])
    stats = {
        m: {
            "mean": float(df[m].mean()),
            "min": float(df[m].min()),
            "max": float(df[m].max()),
            "std": float(df[m].std()),
        }
        for m in df.columns
    }
    return {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "reason": reason,
        **{k: runs[0][k] for k in same},
        "runs": [r["ts"] for r in runs],
        "run_cost_usd": [r["cost_usd"]["total"] for r in runs],
        "metrics": stats,
        "thresholds": {
            m: max(floor, stats[m]["max"] - stats[m]["min"])
            for m, floor in GATE_FLOORS.items()
        },
    }


def write_baseline(reason: str) -> dict:
    """Write data/baseline.json from the last BASELINE_RUNS logged runs. Refuses
    to overwrite an existing baseline: the old one stays in git; delete the file
    to re-baseline on purpose."""
    if BASELINE_PATH.is_file():
        raise FileExistsError(f"{BASELINE_PATH} exists")
    runs = [json.loads(line) for line in SCORES_PATH.open(encoding="utf-8")]
    baseline = make_baseline(runs[-BASELINE_RUNS:], reason)
    BASELINE_PATH.write_text(json.dumps(baseline, indent=1) + "\n", encoding="utf-8")
    return baseline


def log_run(record: dict) -> None:
    """Append one run to SCORES_PATH as a JSON line."""
    with SCORES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_full(label: str) -> tuple[dict, pd.DataFrame]:
    """Full eval: retrieval metrics on 364 queries, answers on the sample, RAGAS
    + judge on the 60 answerable, abstain accuracy on all 80; gate vs
    data/baseline.json ("passed" is None while it is missing); log the run.

    Returns the logged record and the per-query table (for diagnosis).
    """
    load_dotenv(PROJECT_DIR / ".env")
    queries = load_queries()
    sample = make_sample(queries)
    retriever = Retriever()
    client = anthropic.Anthropic()

    ret = retrieval_scores(queries, retriever, client)
    results = run_sample(queries, sample, retriever, client)
    ragas_df, ragas_cost = ragas_scores(results)
    answerable = [r for r in results if r.reference is not None]
    with ThreadPoolExecutor(EVAL_WORKERS) as pool:
        judged = list(
            pool.map(
                lambda r: judge(r.question, r.reference, r.answer, client), answerable
            )
        )
    judge_df = pd.DataFrame([s for s, _ in judged])
    is_answered = [answered(r.answer) for r in answerable]

    per_query = pd.concat(
        [
            pd.DataFrame(
                {
                    "query_id": [r.query_id for r in answerable],
                    "abstained": [r.answer.abstained for r in answerable],
                    "refused": [r.answer.refused for r in answerable],
                    "top1": [r.hits[0].score for r in answerable],
                    "confidence": [r.answer.confidence for r in answerable],
                    "retried": [r.answer.retried for r in answerable],
                    "answer": [r.answer.text for r in answerable],
                }
            ),
            ragas_df.reset_index(drop=True),
            judge_df.add_prefix("judge_"),
        ],
        axis=1,
    )
    metrics = {
        "recall@10": float(ret.recall.mean()),
        "ndcg@10": float(ret.ndcg.mean()),
        **recall_by_content(ret),
        **{c: float(ragas_df[c].mean()) for c in ragas_df.columns},
        **{f"judge_{c}": float(judge_df[c].mean()) for c in JUDGE_CRITERIA},
        **abstain_scores(results),
    }
    cost = {
        "retrieval_eval_search": float(ret.cost.sum()),  # rewrite / HyDE, 364 queries
        "answers": sum(r.answer.cost_usd for r in results),  # incl. their rewrite/HyDE
        "ragas": ragas_cost,
        "judge": sum(c for _, c in judged),
    }
    self_check = None
    if SELF_CHECK:
        self_check, cost["self_check_pairs"] = retry_scores(results, per_query, client)
    cost["total"] = sum(cost.values())

    baseline = load_baseline()
    failed = gate(metrics, baseline) if baseline else []
    record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": label,
        "index": INDEX_NAME,
        "flags": {
            "rewrite": REWRITE,
            "hyde": HYDE,
            "retrieval": RETRIEVAL,
            "rerank": RERANK,
            "self_check": SELF_CHECK,
        },
        "env": run_env(),
        "fingerprint": fingerprint(),
        "metrics": metrics,
        "n_answered": sum(is_answered),
        "n_refused": sum(r.answer.refused for r in results),
        "nan": {  # scores RAGAS failed to compute; abstains are not scored by design
            "faithfulness": int(ragas_df.faithfulness[is_answered].isna().sum()),
            "answer_relevancy": int(
                ragas_df.answer_relevancy[is_answered].isna().sum()
            ),
            "context_precision": int(ragas_df.context_precision.isna().sum()),
            "context_recall": int(ragas_df.context_recall.isna().sum()),
        },
        "cost_usd": cost,
        "self_check": self_check,
        "compared_to": baseline["created"] if baseline else None,
        "failed": failed,
        "passed": not failed if baseline else None,
    }
    log_run(record)
    return record, per_query


def print_run(record: dict) -> None:
    """Metrics, NaN counts, cost and the gate verdict of one run."""
    print(f"{record['ts']}  {record['label']}  ({record['index']})")
    print(f"  flags: {record['flags']}")
    for m, v in record["metrics"].items():
        mark = "  FAILED" if m in record["failed"] else ""
        print(f"  {m:22s} {v:.4f}{mark}")
    print(f"  answered (scored by faithfulness, relevancy): {record['n_answered']}")
    print(f"  refused (not gated): {record['n_refused']}")
    print("  NaN (RAGAS failed):", record["nan"])
    print("  cost $:", {k: round(v, 4) for k, v in record["cost_usd"].items()})
    sc = record.get("self_check")
    if sc:
        print(
            f"  self-check: {sc['n_retried']} of {sc['n_checked']} checked answers "
            f"retried (unanswerable {sc['n_retried_unanswerable']}), answered -> "
            f"abstained {sc['answered_to_abstained']}"
        )
        print("  final confidence counts:", sc["final_confidence_counts"])
        for c, v in sc.get("pairs", {}).items():
            print(f"    {c:22s} first {v['first']:.4f}  final {v['final']:.4f}")
    if record["passed"] is None:
        print("  gate: not gated (no data/baseline.json)")
        return
    same = record["fingerprint"] == load_baseline()["fingerprint"]
    print(f"  code, lockfile and data same as the baseline's: {same}")
    verdict = "PASS" if record["passed"] else f"FAIL {record['failed']}"
    print(f"  gate vs baseline {record['compared_to']}: {verdict}")


def print_retrieval(name: str, ret: pd.DataFrame) -> None:
    """One line: recall@10 and NDCG@10 means over `ret`."""
    print(
        f"{name:28s} n={len(ret):3d}  recall@{EVAL_K} {ret.recall.mean():.4f}"
        f"  NDCG@{EVAL_K} {ret.ndcg.mean():.4f}"
    )


def main() -> None:
    """No argument: free retrieval eval. `followups`: the two-turn tests.
    `full "label"`: the full eval. `baseline "reason"`: write data/baseline.json."""
    logging.basicConfig(level=logging.WARNING)
    if sys.argv[1:2] == ["baseline"]:
        b = write_baseline(sys.argv[2])
        for m, thr in b["thresholds"].items():
            s = b["metrics"][m]
            print(
                f"{m:22s} mean {s['mean']:.4f}  min {s['min']:.4f}"
                f"  max {s['max']:.4f}  threshold {thr:.4f}"
            )
        print(f"wrote {BASELINE_PATH}")
        return
    if sys.argv[1:2] == ["full"]:
        record, _ = run_full(sys.argv[2])
        print_run(record)
        return

    load_dotenv(PROJECT_DIR / ".env")
    client = anthropic.Anthropic()
    if sys.argv[1:2] == ["followups"]:
        conv = conversation_scores(Retriever(), client)
        pd.set_option("display.width", 200, "display.max_colwidth", 90)
        print(conv[["id", "recall raw", "recall rewritten", "recall original"]])
        for _, r in conv.iterrows():
            print(f"{r['id']}: {r['rewritten_query']}")
        for way in ("raw", "rewritten", "original"):
            print(
                f"{way:9s} recall@{EVAL_K} {conv[f'recall {way}'].mean():.4f}"
                f"  NDCG@{EVAL_K} {conv[f'ndcg {way}'].mean():.4f}"
            )
        print(f"rewrite cost ${conv.cost.sum():.4f}")
        return

    queries = load_queries()
    sample = make_sample(queries)
    in_sample = queries["query_id"].isin(sample["answerable"])
    res = retrieval_scores(queries, Retriever(), client)
    keyword = res.format == "keyword"
    n_visual = int(queries.loc[in_sample, "visual"].sum())
    n_many_gold = int((queries.gold.map(len) > EVAL_K).sum())
    print(
        f"index {INDEX_NAME}, page-level @{EVAL_K}, REWRITE={REWRITE} HYDE={HYDE}"
        f" RETRIEVAL={RETRIEVAL} RERANK={RERANK}"
    )
    print_retrieval("all queries", res)
    print_retrieval("non-sample (decides flags)", res[~in_sample])
    print_retrieval("keyword queries", res[keyword])
    print_retrieval("keyword, non-sample", res[keyword & ~in_sample])
    for name, value in recall_by_content(res).items():
        print(f"{name}: {value:.4f}")
    print(f"queries with < {EVAL_K} distinct pages in top chunks: {res.short.sum()}")
    print(f"queries with > {EVAL_K} gold pages: {n_many_gold}")
    p50, p95 = res.search_ms.quantile([0.5, 0.95])
    print(f"search latency: p50 {p50:.0f} ms, p95 {p95:.0f} ms")
    print(
        f"sample: {len(sample['answerable'])} answerable ({n_visual} visual)"
        f" + {len(sample['unanswerable'])} unanswerable"
    )
    print(f"rewrite / HyDE cost ${res.cost.sum():.4f}")


if __name__ == "__main__":
    main()
