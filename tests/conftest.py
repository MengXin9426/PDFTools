"""pytest 共享 fixture：生成含小图标与英文正文的最小 PDF。"""

from __future__ import annotations

import io
from pathlib import Path

import pymupdf
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def minimal_e2e_pdf(tmp_path_factory) -> Path:
    """单页：左上一块 50×50 小图（非整页背景）+ 一段可翻译英文。"""
    d = tmp_path_factory.mktemp("pdf_fixtures")
    path = d / "minimal_e2e.pdf"
    if path.is_file():
        return path

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    im = Image.new("RGB", (50, 50), color=(30, 120, 200))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    rect = pymupdf.Rect(72, 72, 122, 122)
    page.insert_image(rect, stream=buf.getvalue())
    page.insert_text(
        (72, 200),
        "Abstract. This minimal PDF tests the translation pipeline and formula $E=mc^2$.",
        fontsize=11,
    )
    doc.save(str(path))
    doc.close()
    return path
