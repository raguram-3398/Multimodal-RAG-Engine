"""Vision: low-text slide pages -> PNG -> Haiku description -> data/vision.jsonl.

Run once per description set:  python -m multimodal_rag.vision
The descriptions become `kind="vision"` chunks at ingest (`ingest vision`).
"""

import base64
import json
import logging
import statistics
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

from multimodal_rag.config import (
    CLAUDE_MODEL,
    CLAUDE_TIMEOUT_S,
    DOCS_METADATA_PATH,
    EVAL_WORKERS,
    OCR_TIMEOUT_S,
    PDF_DIR,
    PROJECT_DIR,
    VISION_DPI,
    VISION_MAX_CHARS,
    VISION_MAX_TOKENS,
    VISION_PATH,
)
from multimodal_rag.contracts import Chunk
from multimodal_rag.retrieve import cost_usd, load_chunks

log = logging.getLogger(__name__)

DESCRIBE_PROMPT = (
    "This image is one slide from an FDA presentation. Describe it in plain text "
    "so it can be found by search and used to answer questions. Start with the "
    "slide title. Then, for every chart, table, diagram or image: say what it "
    "shows, give axis labels, units and legend entries, and list every number and "
    "label you can read together with what it refers to. Describe only what is "
    "visible; do not interpret it or add outside facts. Skip the FDA logo and "
    "the slide number. No preamble, no markdown."
)


def low_text_pages(
    chunks: dict[str, Chunk], meta: pd.DataFrame
) -> list[tuple[str, int]]:
    """Slide pages (doc_id, 0-indexed page) whose chunk text totals fewer than
    VISION_MAX_CHARS characters. A page with no chunk (empty OCR) counts as 0.
    Book pages are skipped: their low-text pages are title or blank pages."""
    chars: dict[tuple[str, int], int] = defaultdict(int)
    for c in chunks.values():
        if c.kind == "text":
            chars[(c.doc_id, c.page)] += len(c.text)
    return [
        (row.doc_id, p)
        for row in meta.sort_values("doc_id").itertuples()
        if row.doc_type == "slides"
        for p in range(int(row.page_number))
        if chars[(row.doc_id, p)] < VISION_MAX_CHARS
    ]


def render_page(pdf: Path, page: int, dpi: int = VISION_DPI) -> bytes:
    """PNG bytes of 0-indexed `page` (pdftoppm pages are 1-indexed, so it gets
    page + 1). Raises subprocess.TimeoutExpired / CalledProcessError on failure."""
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "page"
        n = str(page + 1)
        subprocess.run(
            ["pdftoppm", "-f", n, "-l", n, "-r", str(dpi), "-png", "-singlefile"]
            + [str(pdf), str(prefix)],
            check=True,
            capture_output=True,
            timeout=OCR_TIMEOUT_S,
        )
        return Path(f"{prefix}.png").read_bytes()


def describe_page(png: bytes, client: anthropic.Anthropic) -> tuple[str | None, float]:
    """Haiku's description of one slide image, and the call's cost.

    The description is None (and logged) when Claude stops with anything but
    `end_turn` (refusal, max_tokens); the page then gets no vision chunk.
    API errors propagate.
    """
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=VISION_MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(png).decode(),
                        },
                    },
                    {"type": "text", "text": DESCRIBE_PROMPT},
                ],
            }
        ],
        timeout=CLAUDE_TIMEOUT_S,
    )
    cost = cost_usd(msg.usage)
    log.info(
        "vision %s in=%d out=%d cost=$%.6f",
        msg.stop_reason,
        msg.usage.input_tokens,
        msg.usage.output_tokens,
        cost,
    )
    if msg.stop_reason != "end_turn":
        return None, cost
    return "".join(b.text for b in msg.content if b.type == "text").strip(), cost


def main() -> None:
    """Describe every low-text slide page and write VISION_PATH (sorted by page)."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    load_dotenv(PROJECT_DIR / ".env")
    meta = pd.read_parquet(DOCS_METADATA_PATH)
    files = dict(zip(meta.doc_id, meta.file_name, strict=True))
    pages = low_text_pages(load_chunks(), meta)
    client = anthropic.Anthropic()

    def one(page: tuple[str, int]) -> tuple[str | None, float]:
        doc_id, p = page
        text, cost = describe_page(render_page(PDF_DIR / files[doc_id], p), client)
        if text is None:
            log.warning("no description for %s page %d", doc_id, p)
        return text, cost

    with ThreadPoolExecutor(EVAL_WORKERS) as pool:
        results = list(pool.map(one, pages))

    records = [
        {"doc_id": d, "page": p, "text": t, "cost_usd": c}
        for (d, p), (t, c) in zip(pages, results, strict=True)
        if t is not None
    ]
    with VISION_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lengths = [len(r["text"]) for r in records]
    print(f"pages: {len(pages)}  described: {len(records)}")
    print(f"chars per description: median {statistics.median(lengths):.0f}")
    print(f"cost: ${sum(c for _, c in results):.4f}")


if __name__ == "__main__":
    main()
