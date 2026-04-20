"""chunk 与内嵌图合并（无 API）。"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from skills.chunk_figure_merge import merge_embedded_figures_into_chunks
from skills.image_extractor import ImageExtractor
from skills.ocr_to_chunks import parse_ocr_to_chunks


def test_merge_inserts_image_chunk_for_small_embed(tmp_path, minimal_e2e_pdf: Path):
    md = """## Page 1

Abstract. This minimal PDF tests the translation pipeline and formula $E=mc^2$.
"""
    chunks = parse_ocr_to_chunks(md)
    assert not any(c["type"] == "image" for c in chunks)

    out_root = tmp_path / "out"
    out_root.mkdir()
    ext = ImageExtractor(str(out_root))
    stem = "minimal_e2e"
    images = ext.extract(str(minimal_e2e_pdf), stem, pages=[1])
    assert len(images) >= 1

    merged = merge_embedded_figures_into_chunks(chunks, images, str(minimal_e2e_pdf))
    assert any(c["type"] == "image" for c in merged)
    img_chunks = [c for c in merged if c["type"] == "image"]
    assert img_chunks[0]["page"] == 1
    assert Path(images[0]["path"]).is_file()


def test_full_page_background_skipped(tmp_path):
    """整页单图视为背景，不插入 image chunk。"""
    path = tmp_path / "fullbg.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)
    big = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 1800, 1800), 1)
    big.clear_with(240)
    r = page.rect
    page.insert_image(r, pixmap=big)
    doc.save(str(path))
    doc.close()

    md = "## Page 1\n\nSome text only.\n"
    chunks = parse_ocr_to_chunks(md)
    ext = ImageExtractor(str(tmp_path / "imgout"))
    images = ext.extract(str(path), "fullbg", pages=[1])
    assert len(images) == 1
    merged = merge_embedded_figures_into_chunks(
        chunks, images, str(path), max_page_coverage=0.88
    )
    assert not any(c["type"] == "image" for c in merged)
