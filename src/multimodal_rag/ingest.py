"""Ingest: 52 FDA PDFs -> Page -> Chunk (+ vision chunks) -> chunks.jsonl -> Pinecone.

Full build from the PDFs:  python -m multimodal_rag.ingest
New index version from the committed text chunks + data/vision.jsonl, without
re-reading the PDFs (keeps text byte-identical to the previous index):
    python -m multimodal_rag.ingest vision
"""

import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element
from unstructured.nlp.tokenize import sent_tokenize
from unstructured.partition.pdf import partition_pdf

from multimodal_rag.config import (
    BOOK_MAX_CHARS,
    CHUNKS_PATH,
    CLOUD,
    DOCS_METADATA_PATH,
    EMBED_BATCH_SIZE,
    EMBED_DIM,
    EMBED_MODEL,
    FRESHNESS_TIMEOUT_S,
    INDEX_NAME,
    INDEX_READY_TIMEOUT_S,
    LOW_TEXT_CHARS,
    MAX_TOKENS,
    METRIC,
    NAMESPACE,
    OCR_DPI,
    OCR_TIMEOUT_S,
    PDF_DIR,
    PINECONE_TIMEOUT_S,
    PROJECT_DIR,
    REGION,
    UPSERT_BATCH_SIZE,
    VISION_PATH,
)
from multimodal_rag.contracts import Chunk, Page

log = logging.getLogger(__name__)


class IngestError(Exception):
    """Input does not match what the pipeline relies on (missing file, page count)."""


def ocr_page(pdf: Path, page: int, timeout_s: float = OCR_TIMEOUT_S) -> str:
    """OCR one 0-indexed page: render it with pdftoppm, read it with tesseract.

    Runs as subprocesses so the timeout really kills a stuck page.
    Raises subprocess.TimeoutExpired / CalledProcessError on failure.
    """
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "page"
        n = str(page + 1)  # pdftoppm pages are 1-indexed
        subprocess.run(
            ["pdftoppm", "-f", n, "-l", n, "-r", str(OCR_DPI), "-png", "-singlefile"]
            + [str(pdf), str(prefix)],
            check=True,
            capture_output=True,
            timeout=timeout_s,
        )
        result = subprocess.run(
            ["tesseract", f"{prefix}.png", "-", "-l", "eng"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    return result.stdout.strip()


def _page_pdf(reader: PdfReader, page: int) -> io.BytesIO:
    """One page of `reader` as an in-memory single-page PDF."""
    writer = PdfWriter()
    writer.add_page(reader.pages[page])
    buf = io.BytesIO()
    writer.write(buf)
    buf.seek(0)
    return buf


def load_pdf(pdf: Path, n_pages: int) -> list[tuple[Page, list[Element]]]:
    """Return every page of `pdf` in order (0-indexed), with its `fast` elements.

    Each page is partitioned as its own 1-page PDF: unstructured's "too complex"
    check otherwise drops a whole PDF to hi_res with no text when any one page has
    heavy vector graphics (true for 2 of the 52 files). A page with fewer than
    LOW_TEXT_CHARS characters is OCR'd instead and gets no elements. A page whose
    OCR fails or times out keeps empty text (it yields no chunk) and is logged.
    Raises IngestError if the file is missing or its page count differs from
    the metadata.
    """
    if not pdf.is_file():
        raise IngestError(f"missing PDF: {pdf}")
    reader = PdfReader(pdf)
    if len(reader.pages) != n_pages:
        raise IngestError(f"{pdf.name}: {len(reader.pages)} pages, metadata {n_pages}")

    out: list[tuple[Page, list[Element]]] = []
    for p in range(n_pages):
        elements = [
            e
            for e in partition_pdf(
                file=_page_pdf(reader, p), strategy="fast", languages=["eng"]
            )
            if e.text.strip()
        ]
        text = "\n\n".join(e.text for e in elements)
        if len(text) >= LOW_TEXT_CHARS:
            out.append((Page(pdf.stem, p, text, ocr=False), elements))
            continue
        try:
            text = ocr_page(pdf, p)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            log.warning("OCR failed for %s page %d: %s", pdf.stem, p, e)
            text = ""
        out.append((Page(pdf.stem, p, text, ocr=True), []))
    return out


def split_to_limit(
    text: str, count: Callable[[str], int], limit: int = MAX_TOKENS
) -> list[str]:
    """Split `text` into pieces of <= `limit` tokens.

    Packs whole sentences greedily; a single sentence over `limit` is packed
    word by word. Token counts add up across pieces joined by a space because
    BERT tokenization splits on whitespace first.
    """
    pieces: list[str] = []
    current: list[str] = []
    used = 0

    def flush() -> None:
        nonlocal current, used
        if current:
            pieces.append(" ".join(current))
        current, used = [], 0

    for sentence in sent_tokenize(text):
        units = [sentence] if count(sentence) <= limit else sentence.split()
        for unit in units:
            n = count(unit)
            if used + n > limit:
                flush()
            current.append(unit)
            used += n
    flush()
    return pieces


def chunk_page(
    page: Page, elements: list[Element], is_book: bool, count: Callable[[str], int]
) -> list[Chunk]:
    """Chunk one page; every chunk has <= MAX_TOKENS tokens and this page's id.

    Candidates: the whole page for slides (and for OCR'd pages, which have no
    elements); `chunk_by_title` over the page's elements for book pages. Any
    candidate over MAX_TOKENS is split by `split_to_limit`.
    """
    if not page.text.strip():
        return []
    if is_book and elements:
        candidates = [
            c.text for c in chunk_by_title(elements, max_characters=BOOK_MAX_CHARS)
        ]
    else:
        candidates = [page.text]

    texts: list[str] = []
    for cand in candidates:
        texts.extend(
            [cand] if count(cand) <= MAX_TOKENS else split_to_limit(cand, count)
        )
    return [
        Chunk(
            f"{page.doc_id}-{page.page}-{n}",
            page.doc_id,
            page.page,
            t,
            count(t),
            "text",
        )
        for n, t in enumerate(texts)
    ]


def vision_chunks(
    descriptions: list[dict], text_chunks: list[Chunk], count: Callable[[str], int]
) -> list[Chunk]:
    """`kind="vision"` chunks for `descriptions` ({doc_id, page, text}, page
    0-indexed), on the same page as the description.

    Chunk numbers continue after the page's text chunks (`doc_id-page-n`), so ids
    stay deterministic. A description over MAX_TOKENS is split like page text.
    """
    n_text: dict[tuple[str, int], int] = {}
    for c in text_chunks:
        n_text[(c.doc_id, c.page)] = n_text.get((c.doc_id, c.page), 0) + 1
    out: list[Chunk] = []
    for d in descriptions:
        doc_id, page, text = d["doc_id"], d["page"], d["text"]
        pieces = [text] if count(text) <= MAX_TOKENS else split_to_limit(text, count)
        first = n_text.get((doc_id, page), 0)
        out.extend(
            Chunk(f"{doc_id}-{page}-{first + i}", doc_id, page, t, count(t), "vision")
            for i, t in enumerate(pieces)
        )
    return out


def with_vision(text_chunks: list[Chunk], count: Callable[[str], int]) -> list[Chunk]:
    """`text_chunks` followed by the vision chunks built from VISION_PATH."""
    with VISION_PATH.open(encoding="utf-8") as f:
        descriptions = [json.loads(line) for line in f]
    return text_chunks + vision_chunks(descriptions, text_chunks, count)


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    """Write one JSON object per chunk (the `Chunk` fields), in chunk order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def upsert(chunks: list[Chunk]) -> int:
    """Embed `chunks` with BGE and upsert them to INDEX_NAME / NAMESPACE.

    Creates the serverless index if missing. Returns the namespace vector count
    once it equals len(chunks); raises IngestError if it does not within
    FRESHNESS_TIMEOUT_S (Pinecone counts are eventually consistent).
    """
    from pinecone import Pinecone, ServerlessSpec
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    vectors = model.encode(
        [c.text for c in chunks],  # passages take no query prefix
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric=METRIC,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
            timeout=INDEX_READY_TIMEOUT_S,
        )
    index = pc.Index(INDEX_NAME)
    records = [
        {
            "id": c.chunk_id,
            "values": v.tolist(),
            "metadata": {
                "doc_id": c.doc_id,
                "page": c.page,
                "text": c.text,
                "kind": c.kind,
                "embedding_model": EMBED_MODEL,
            },
        }
        for c, v in zip(chunks, vectors, strict=True)
    ]
    for i in range(0, len(records), UPSERT_BATCH_SIZE):
        index.upsert(
            vectors=records[i : i + UPSERT_BATCH_SIZE],
            namespace=NAMESPACE,
            timeout=PINECONE_TIMEOUT_S,
        )

    deadline = time.monotonic() + FRESHNESS_TIMEOUT_S
    while True:
        stats = index.describe_index_stats(timeout=PINECONE_TIMEOUT_S)
        ns = stats.namespaces.get(NAMESPACE)
        count = int(ns.vector_count) if ns else 0
        if count == len(chunks):
            return count
        if time.monotonic() > deadline:
            raise IngestError(f"{count}/{len(chunks)} vectors visible in {NAMESPACE}")
        time.sleep(2)


def main() -> None:
    """Build chunks (from the PDFs, or `vision`: from the committed text chunks),
    add the vision chunks, write chunks.jsonl, embed + upsert, print a summary."""
    from transformers import AutoTokenizer

    from multimodal_rag.retrieve import load_chunks

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv(PROJECT_DIR / ".env")
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)

    def count(text: str) -> int:
        return len(tok(text, add_special_tokens=False, verbose=False)["input_ids"])

    if sys.argv[1:2] == ["vision"]:
        text = [c for c in load_chunks().values() if c.kind == "text"]
        chunks = with_vision(text, count)
        write_chunks(chunks, CHUNKS_PATH)
        print(f"text chunks: {len(text)}  vision chunks: {len(chunks) - len(text)}")
        print(f"max tokens: {max(c.n_tokens for c in chunks)}")
        print(f"vectors in {INDEX_NAME}/{NAMESPACE}: {upsert(chunks)}")
        return

    meta = pd.read_parquet(DOCS_METADATA_PATH)
    t0 = time.monotonic()
    pages: list[Page] = []
    chunks: list[Chunk] = []
    for row in meta.sort_values("doc_id").itertuples():
        loaded = load_pdf(PDF_DIR / row.file_name, int(row.page_number))
        for page, elements in loaded:
            pages.append(page)
            chunks.extend(chunk_page(page, elements, row.doc_type == "book", count))
        log.info("%s: %d pages", row.doc_id, len(loaded))
    chunks = with_vision(chunks, count)
    write_chunks(chunks, CHUNKS_PATH)

    ocr = [p for p in pages if p.ocr]
    empty = sum(not p.text for p in ocr)
    print(f"pages: {len(pages)}  ocr: {len(ocr)}  ocr empty: {empty}")
    print(f"chunks: {len(chunks)}  max tokens: {max(c.n_tokens for c in chunks)}")
    print(f"load + chunk: {time.monotonic() - t0:.0f} s")
    print(f"vectors in {INDEX_NAME}/{NAMESPACE}: {upsert(chunks)}")


if __name__ == "__main__":
    main()
