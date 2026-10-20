"""Vision edge cases: the rendered image is the requested 0-indexed page, and
page selection counts pages that produced no chunk. No network calls."""

import io

import pandas as pd
from pypdf import PdfWriter

from multimodal_rag.contracts import Chunk
from multimodal_rag.vision import low_text_pages, render_page


def test_render_page_is_0_indexed(tmp_path) -> None:
    """Pages of different widths: page 1 (0-indexed) must come out 200 pt wide."""
    writer = PdfWriter()
    for width in (100, 200, 300):
        writer.add_blank_page(width=width, height=100)
    pdf = tmp_path / "three.pdf"
    buf = io.BytesIO()
    writer.write(buf)
    pdf.write_bytes(buf.getvalue())
    png = render_page(pdf, 1, dpi=72)  # 72 dpi: 1 pt = 1 px
    width = int.from_bytes(png[16:20], "big")  # PNG IHDR width
    assert width == 200


def test_low_text_pages_include_pages_without_chunks_and_skip_books() -> None:
    chunks = {
        "deck-0-0": Chunk("deck-0-0", "deck", 0, "x" * 400, 90, "text"),
        "deck-1-0": Chunk("deck-1-0", "deck", 1, "Title only", 3, "text"),
        "book-0-0": Chunk("book-0-0", "book", 0, "short", 1, "text"),
    }
    meta = pd.DataFrame(
        {
            "doc_id": ["deck", "book"],
            "doc_type": ["slides", "book"],
            "page_number": [3, 2],
        }
    )
    # page 2 of the deck has no chunk at all (empty OCR)
    assert low_text_pages(chunks, meta) == [("deck", 1), ("deck", 2)]
