"""完整流程端到端测试（OCR API + 翻译 API + 本地插图 + LaTeX PDF）。

运行方式：
  pip install -r requirements-dev.txt
  export QWEN_API_KEY=sk-...
  pytest tests/test_e2e_full_pipeline.py -m e2e_api -s --tb=short

未设置 ``QWEN_API_KEY`` 时自动跳过，避免 CI 无密钥失败。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skills.load_api_config import qwen_api_key
from skills.qwen_vl_ocr import extract_pdf_to_markdown
from skills.ocr_to_chunks import parse_ocr_to_chunks
from skills.chunk_figure_merge import merge_embedded_figures_into_chunks
from skills.image_extractor import ImageExtractor
from skills.translator.qwen_translator import QwenTranslator
from skills.latex_builder import LatexBuilder
from skills.latex_builder.plaintext import write_translated_plaintext


pytestmark = pytest.mark.e2e_api


def _have_qwen_key() -> bool:
    return bool((os.environ.get("QWEN_API_KEY") or "").strip() or qwen_api_key())


@pytest.mark.skipif(not _have_qwen_key(), reason="需要 QWEN_API_KEY 或 config.yaml 中的 qwen.api_key")
def test_full_pipeline_ocr_translate_latex_pdf(tmp_path, minimal_e2e_pdf: Path):
    """Qwen-VL-OCR → chunk → 合并插图 → Qwen 翻译 → xelatex 生成 PDF。"""
    stem = minimal_e2e_pdf.stem
    output_dir = tmp_path / "pipeline_out"
    output_dir.mkdir(parents=True)

    en_md = extract_pdf_to_markdown(
        minimal_e2e_pdf,
        output_dir=output_dir / "ocr",
        dpi=120,
        model="qwen-vl-ocr",
        pages=[1],
    )
    assert len(en_md) > 20
    (output_dir / "english.md").write_text(en_md, encoding="utf-8")

    chunks = parse_ocr_to_chunks(en_md)
    assert any(c["type"] == "text_group" for c in chunks)

    img_root = tmp_path / "img_root"
    img_root.mkdir()
    extractor = ImageExtractor(str(img_root))
    images = extractor.extract(str(minimal_e2e_pdf), stem, pages=[1])
    assert len(images) >= 1

    chunks = merge_embedded_figures_into_chunks(chunks, images, str(minimal_e2e_pdf))

    api_key = qwen_api_key()
    assert api_key
    translator = QwenTranslator(api_key=api_key, model=os.environ.get("PDFTOOLS_TRANSLATE_MODEL", "qwen-plus"))
    translated = translator.translate_chunks(chunks)
    assert translated and any(c.get("type") == "text_group" for c in translated)

    zh_txt = output_dir / "translated_zh.txt"
    write_translated_plaintext(translated, zh_txt)
    text_out = zh_txt.read_text(encoding="utf-8")
    assert len(text_out.strip()) > 5

    latex_out = tmp_path / "latex_out"
    (latex_out / "result" / stem / "images").mkdir(parents=True)
    shutil.copytree(
        img_root / "result" / stem / "images",
        latex_out / "result" / stem / "images",
        dirs_exist_ok=True,
    )

    builder = LatexBuilder(str(latex_out))
    builder.build_latex(stem, translated, images)
    inner_tex = latex_out / "result" / stem / "document.tex"
    inner_pdf = latex_out / "result" / stem / "document.pdf"
    assert inner_tex.is_file(), "应写出 LaTeX 源文件"
    assert inner_pdf.is_file(), "应在 xelatex 可用时生成 PDF（参见 skills/latex_builder 退出码处理）"
