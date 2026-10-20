"""Constants for the Multimodal RAG Engine. Grows day by day. The only runtime
options are the measured ablation flags (environment variables, README: Flags)."""

import os
from pathlib import Path

PROJECT_DIR: Path = Path(__file__).resolve().parents[2]
DATASET_DIR: Path = PROJECT_DIR / "datasets" / "vidore_fda"  # gitignored
PDF_DIR: Path = DATASET_DIR / "pdfs"
DOCS_METADATA_PATH: Path = (
    DATASET_DIR / "documents_metadata" / "test-00000-of-00001.parquet"
)
CORPUS_INDEX_PATH: Path = DATASET_DIR / "corpus_index.parquet"
DATA_DIR: Path = PROJECT_DIR / "data"
CHUNKS_PATH: Path = DATA_DIR / "chunks.jsonl"

# Loading
LOW_TEXT_CHARS: int = 50  # pages with less extracted text than this get OCR
OCR_DPI: int = 200  # same render resolution unstructured uses for OCR
OCR_TIMEOUT_S: float = 60.0  # per subprocess (render, then tesseract); ~3 s typical

# Chunking
BOOK_MAX_CHARS: int = 1500  # chunk_by_title hard limit for book pages
MAX_TOKENS: int = 510  # BGE's 512 minus [CLS] and [SEP]

# Embedding
EMBED_MODEL: str = "BAAI/bge-small-en-v1.5"
EMBED_DIM: int = 384
EMBED_BATCH_SIZE: int = 32

# Pinecone
INDEX_NAME: str = "fda-bge-small-en-v1-5-v2"  # bump the suffix on every re-index
NAMESPACE: str = "fda"
METRIC: str = "cosine"
CLOUD: str = "aws"
REGION: str = "us-east-1"  # the free Starter plan only offers aws/us-east-1
PINECONE_TIMEOUT_S: float = 10.0  # every single Pinecone request
INDEX_READY_TIMEOUT_S: int = 120  # create_index waits this long for "Ready"
FRESHNESS_TIMEOUT_S: float = 60.0  # wait for upserts to show in the vector count
UPSERT_BATCH_SIZE: int = 100

# Eval data
QUERIES_PATH: Path = DATASET_DIR / "queries" / "test-00000-of-00001.parquet"
QRELS_PATH: Path = DATASET_DIR / "qrels" / "test-00000-of-00001.parquet"
EVAL_DIR: Path = DATA_DIR / "eval"
UNANSWERABLE_PATH: Path = EVAL_DIR / "unanswerable_en.jsonl"
SAMPLE_PATH: Path = EVAL_DIR / "sample.json"  # fixed once on Day 28, never changed
SAMPLE_SIZE: int = 60  # answerable queries; plus all 20 unanswerable
SAMPLE_SEED: int = 28
VISUAL_CONTENT: frozenset[str] = frozenset({"Chart", "Infographic", "Table", "Image"})

# Retrieval
# BGE v1.5 instruction for short query -> passage retrieval (passages take none)
QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "
TOP_K: int = 20  # chunks fetched from Pinecone (also the rerank budget)
EVAL_K: int = 10  # distinct pages scored by recall@10 / NDCG@10

# Query transforms (Day 31). REWRITE=1 / HYDE=1 in the environment turn them on.
# A follow-up (non-empty chat history) is always rewritten, flag or not.
REWRITE: bool = os.environ.get("REWRITE") == "1"
HYDE: bool = os.environ.get("HYDE") == "1"
REWRITE_MAX_TOKENS: int = 1024  # forced tool call, no thinking; one line
HYDE_MAX_TOKENS: int = 2048  # thinking counts; the passage is ~100 words
FOLLOWUPS_PATH: Path = EVAL_DIR / "followups.jsonl"  # 10 two-turn tests

# Hybrid retrieval (Day 32): BM25 and vector search fused by RRF. Default since
# Ram's Mac run passed the gate (2026-10-12); RETRIEVAL=vector for vector only.
RETRIEVAL: str = os.environ.get("RETRIEVAL", "hybrid")  # "vector" or "hybrid"
RRF_K: int = 60  # RRF constant from Cormack et al. (2009); not tuned
# rank_bm25's BM25Okapi floors a negative IDF (term in > half the chunks) at
# epsilon x mean IDF. Its default 0.25 gave "the", "of", "and", "in", "for", "a",
# "to" IDF 1.79, above "fda" (1.21) and "drug" (0.97). 0 scores them 0 (Day 32).
BM25_EPSILON: float = 0.0

# Reranker (Day 33): a cross-encoder re-scores the TOP_K retrieved chunks against
# the question, best first. Default since Ram's Mac run passed the gate with the
# version/year abstain rule (2026-10-13); RERANK=0 turns it off.
RERANK: bool = os.environ.get("RERANK", "1") == "1"
RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L6-v2"  # 512-token pair limit

# Answer
CLAUDE_MODEL: str = "claude-haiku-5-5"  # never pass `temperature` (400 on 5.5)
CLAUDE_TIMEOUT_S: float = 30.0  # per attempt; the SDK retries twice on its own
ANSWER_K: int = 10  # top chunks sent to Claude as documents
ANSWER_MAX_TOKENS: int = (
    4096  # 2048 cut one Day 29 eval answer (1,024 cut 2/8 on Day 28)
)
PRICE_IN_PER_MTOK: float = 0.10  # Haiku 5.5, prompts <= 100k tokens (pricing page)
PRICE_OUT_PER_MTOK: float = 0.50
ABSTAIN_TEXT: str = "The documents do not contain the answer."
REFUSAL_TEXT: str = "The model declined to answer this question."

# Self-check (Day 34): Haiku scores each answer 1-10; below SELF_CHECK_MIN, one
# more search with a query rewritten from the stated gap, then the final answer.
# SELF_CHECK=1 turns it on.
SELF_CHECK: bool = os.environ.get("SELF_CHECK") == "1"
SELF_CHECK_MIN: int = 6  # from the plan; not tuned
CHECK_MAX_TOKENS: int = 1024  # forced tool call, no thinking; a score + a sentence

# Vision descriptions (Day 30)
VISION_PATH: Path = DATA_DIR / "vision.jsonl"  # one description per low-text slide
VISION_MAX_CHARS: int = 300  # slide pages with less chunk text get a description
# 100 dpi = 1000x750 px per slide (~970 visual tokens); on 3 measured slides the
# descriptions read every label and value as well as at 150 dpi
VISION_DPI: int = 100
VISION_MAX_TOKENS: int = 8192  # thinking counts; measured max 1,581 output tokens
VISION_CONTENT: tuple[str, ...] = ("Chart", "Infographic", "Image")  # recall split

# Eval pipeline (Day 29)
SCORES_PATH: Path = DATA_DIR / "scores.jsonl"  # one line per full eval run
EVAL_WORKERS: int = 8  # parallel Claude calls for answers and the judge
# Haiku 5.5 thinks by default; thinking tokens count against max_tokens (and are
# billed as output). At 4096, 13/60 faithfulness verdicts on the longest answers
# were cut off (2.7k of 4,096 output tokens were thinking).
RAGAS_MAX_TOKENS: int = 16384
JUDGE_MAX_TOKENS: int = 512
# Gate: a run fails if a gated metric is below the baseline mean by more than
# its threshold. data/baseline.json holds the mean of BASELINE_RUNS unchanged full
# runs on Ram's Mac; threshold = max(floor, max - min of those runs). Floors: 0.05
# for faithfulness and answer relevancy; 0.005 for recall@10 and NDCG@10 (no
# Claude calls without REWRITE/HYDE: identical on every Mac run since Day 29); 0
# for unanswerable abstain (no drop beyond the baseline runs' own spread); 2/60
# for answerable answered (the same unchanged code flipped 2 queries, 152 and
# 212, in an earlier sandbox pair).
BASELINE_PATH: Path = DATA_DIR / "baseline.json"
BASELINE_RUNS: int = 5
GATE_FLOORS: dict[str, float] = {
    "faithfulness": 0.05,
    "answer_relevancy": 0.05,
    "recall@10": 0.005,
    "ndcg@10": 0.005,
    "abstain_unanswerable": 0.0,
    "answer_answerable": 2 / 60,
}

# App (Day 35): one Streamlit page; visitors bring their own Anthropic key.
# Questions per browser session (resets on page reload). Visitors pay for
# Claude; this limits Pinecone free-tier requests and the 2 shared vCPUs.
SESSION_QUERY_CAP: int = 20
# Non-sample queries 243, 255, 254, 197 (answerable) and dev trap dm01 (abstains)
EXAMPLE_QUESTIONS: tuple[str, ...] = (
    "What kind of consumer research is the FDA conducting for naloxone’s OTC "
    "transition?",
    "ANDAs received by FDA in FY12",
    "List which organizations collaborated with CDER on data harmonization for RWE.",
    "Explain how did the FDA's refuse to receive percentage in FY 16 compare to "
    "FY 15.",
    "What percentage of ANDAs were refused to receive (RTR) in FY2020?",
)
