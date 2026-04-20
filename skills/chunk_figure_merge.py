"""将 PDF 内嵌图与 OCR chunk 对齐：OCR 常缺少 \\begin{figure}，此处按页补 image chunk。

整页位图（版心背景）通过版面占比过滤，避免把每页大图当作插图重复插入。
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import pymupdf


def _chunk_page(chunk: Dict) -> int:
    if chunk["type"] == "text_group":
        return int(chunk["chunks"][0].get("page", 0))
    return int(chunk.get("page", 0))


def _page_area(page: pymupdf.Page) -> float:
    r = page.rect
    return max(r.width * r.height, 1.0)


def _image_coverage_ratio(doc: pymupdf.Document, page_num: int, xref: int) -> float:
    """返回该 xref 在页面上的覆盖面积 / 页面面积（多矩形时求和）。"""
    page = doc[page_num - 1]
    rects = page.get_image_rects(xref)
    if not rects:
        return 0.0
    covered = sum(max(r.width * r.height, 0.0) for r in rects)
    return min(covered / _page_area(page), 1.0)


def merge_embedded_figures_into_chunks(
    chunks: List[Dict],
    images: List[Dict],
    pdf_path: str,
    *,
    max_page_coverage: float = 0.88,
) -> List[Dict]:
    """在 OCR chunk 序列中插入「有内嵌图但 OCR 未产出 figure 环境」的 image chunk。

    - 若某页已有来自 OCR 的 ``type == "image"`` chunk，则不再为该页自动补图（避免重复）。
    - ``max_page_coverage``：单图在单页上的覆盖比例超过该阈值时视为整页背景，跳过。
    """
    if not images:
        return chunks

    doc = pymupdf.open(pdf_path)
    try:
        pages_with_ocr_figure: Set[int] = {
            int(c["page"])
            for c in chunks
            if c.get("type") == "image" and c.get("page")
        }

        extras: List[Tuple[int, int]] = []  # (page, figure_slot index 1-based)
        for info in images:
            page = int(info["page"])
            xref = int(info["xref"])
            if page in pages_with_ocr_figure:
                continue
            cov = _image_coverage_ratio(doc, page, xref)
            if cov >= max_page_coverage:
                continue
            slot = int(info.get("index", 1))
            extras.append((page, slot))
    finally:
        doc.close()

    if not extras:
        return chunks

    extras_by_page: Dict[int, List[int]] = {}
    for p, slot in extras:
        extras_by_page.setdefault(p, []).append(slot)

    out: List[Dict] = []
    i = 0
    n = len(chunks)
    seen_pages: Set[int] = set()

    while i < n:
        p0 = _chunk_page(chunks[i])
        seen_pages.add(p0)
        j = i
        while j < n and _chunk_page(chunks[j]) == p0:
            out.append(chunks[j])
            j += 1
        for slot in extras_by_page.get(p0, []):
            out.append(
                {
                    "type": "image",
                    "page": p0,
                    "figure_slot": slot,
                    "bbox": (0, 0, 0, 0),
                }
            )
        i = j

    for p, slots in extras_by_page.items():
        if p in seen_pages:
            continue
        for slot in slots:
            out.append(
                {
                    "type": "image",
                    "page": p,
                    "figure_slot": slot,
                    "bbox": (0, 0, 0, 0),
                }
            )

    return out
