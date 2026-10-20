"""Stage contracts: frozen dataclasses passed between pipeline stages.

Kept apart from `ingest.py` so the query path (retrieve, answer, UI) never
imports `unstructured`, spaCy or tesseract.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    """One PDF page. `page` is 0-indexed (dataset convention); `ocr` = text from OCR."""

    doc_id: str
    page: int
    text: str
    ocr: bool


@dataclass(frozen=True)
class Chunk:
    """One embeddable unit: <= MAX_TOKENS BGE tokens, never crosses a page.

    `chunk_id` is `doc_id-page-n` (n = 0, 1, ... within the page), so re-running
    ingest overwrites the same Pinecone records instead of duplicating them.
    """

    chunk_id: str
    doc_id: str
    page: int
    text: str
    n_tokens: int
    kind: str  # "text" (PDF text) or "vision" (generated description)


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk and its retrieval score: cosine similarity (vector),
    BM25 score (`bm25_search`), RRF score (hybrid) or cross-encoder
    logit (RERANK=1)."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class SearchQuery:
    """What retrieval embeds for one user question.

    `question` is what Claude answers: the user's question, or for a follow-up
    its standalone rewrite (`answer` takes no chat history). `text` is what BGE
    embeds; `passage` = embed it as a passage (HyDE), not as a query.
    `cost_usd` is the rewrite and HyDE calls, 0 when neither ran.
    """

    question: str
    text: str
    passage: bool
    cost_usd: float


@dataclass(frozen=True)
class Citation:
    """A verified quote: `cited_text` is exactly `chunk.text[start:end]` of `chunk_id`.

    `page` is 0-indexed. `score` is the cited chunk's `Hit.score`
    (cosine, RRF or cross-encoder logit, by the flags).
    `generated` marks quotes from generated vision descriptions.
    """

    chunk_id: str
    doc_id: str
    page: int
    cited_text: str
    score: float
    generated: bool


@dataclass(frozen=True)
class Answer:
    """What the pipeline returns for one question.

    `abstained` = no answer from the documents (no hits, or Claude replied with
    ABSTAIN_TEXT and cited nothing). `refused` = Claude
    stopped with `refusal`: `text` is REFUSAL_TEXT, no citations, not an abstain.
    `cost_usd` is 0 when Claude was not called.
    `confidence` = Haiku's 1-10 self-check score of this answer; None when not
    checked (SELF_CHECK off, abstain, refusal). `retried` = the first answer
    scored below SELF_CHECK_MIN, so this is the answer after one more search.
    """

    text: str
    citations: list[Citation]
    abstained: bool
    refused: bool
    cost_usd: float
    latency_s: float
    confidence: int | None = None
    retried: bool = False
