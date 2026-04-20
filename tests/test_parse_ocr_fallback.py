"""OCR Markdown 解析兜底（无 API）。"""

from skills.ocr_to_chunks import parse_ocr_to_chunks


def test_parse_without_page_markers_treated_as_page1():
    md = "No page header here.\n\nJust **markdown** and $x=1$.\n"
    chunks = parse_ocr_to_chunks(md)
    assert chunks, "无 ## Page 时仍应得到 chunk"
    assert any(c["type"] == "text_group" for c in chunks)
    tg = next(c for c in chunks if c["type"] == "text_group")
    assert tg["chunks"][0]["page"] == 1


def test_parse_all_decorator_lines_still_yield_text_via_fallback():
    """若可解析行均被跳过，仍应用整页原文生成 text_group（避免下游空输出）。"""
    md = """## Page 1

\\begin{center}
\\end{center}
"""
    chunks = parse_ocr_to_chunks(md)
    assert any(c["type"] == "text_group" for c in chunks)
    blob = "\n".join(
        line
        for c in chunks
        if c["type"] == "text_group"
        for sub in c.get("chunks", [])
        for line in sub.get("content", [])
    )
    assert "center" in blob
