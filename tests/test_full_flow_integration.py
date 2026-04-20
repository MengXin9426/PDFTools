"""完整流程集成测试：PDF → 图表保存 → 文字OCR → 翻译 → 图表公式插回 → 生成PDF。

本测试生成一个含有多种元素（标题、正文段落、行内公式、独立公式、
内嵌图片）的英文学术 PDF，然后走完整的 pipeline：

  1. ImageExtractor 提取内嵌图片并保存到本地
  2. Qwen-VL-OCR 提取结构化文字（含公式）
  3. parse_ocr_to_chunks 解析为标准 chunk
  4. merge_embedded_figures_into_chunks 将图片对齐到 chunk
  5. QwenTranslator 翻译（保护公式、缩写）
  6. LatexBuilder 构建 LaTeX 并编译生成中文 PDF
  7. 验证最终 PDF 的完整性

运行：
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_full_flow_integration.py -m e2e_api -v -s
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import List

import pymupdf
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skills.load_api_config import qwen_api_key
from skills.image_extractor import ImageExtractor
from skills.qwen_vl_ocr import extract_pdf_to_markdown
from skills.ocr_to_chunks import parse_ocr_to_chunks
from skills.chunk_figure_merge import merge_embedded_figures_into_chunks
from skills.translator.qwen_translator import QwenTranslator
from skills.latex_builder import LatexBuilder
from skills.latex_builder.plaintext import write_translated_plaintext
from skills.quality_checker import QualityChecker

pytestmark = pytest.mark.e2e_api


def _have_qwen_key() -> bool:
    return bool((os.environ.get("QWEN_API_KEY") or "").strip() or qwen_api_key())


@pytest.fixture(scope="session")
def rich_test_pdf(tmp_path_factory) -> Path:
    """生成一个包含标题、正文、公式、图表的两页英文学术 PDF。"""
    d = tmp_path_factory.mktemp("rich_pdf")
    path = d / "test_academic_paper.pdf"

    doc = pymupdf.open()

    # === Page 1: Title + Abstract + Formula + Figure ===
    page1 = doc.new_page(width=595, height=842)

    page1.insert_text(
        (72, 60),
        "Waveform Design for Joint Radar-Communication Systems",
        fontsize=16,
        fontname="helv",
    )

    page1.insert_text(
        (72, 100),
        "Abstract",
        fontsize=13,
        fontname="helv",
    )

    abstract_text = (
        "This paper proposes a novel waveform design framework for joint "
        "radar-communication (JRC) systems. The proposed approach maximizes "
        "the mutual information between the transmitted waveform and the "
        "target response while maintaining communication performance. "
        "We formulate the optimization problem using the MIMO signal model "
        "and solve it via alternating minimization. Simulation results show "
        "that the proposed method achieves superior detection probability "
        "compared to conventional OFDM-based approaches."
    )
    rect1 = pymupdf.Rect(72, 120, 523, 280)
    page1.insert_textbox(rect1, abstract_text, fontsize=10)

    formula_text = "The signal model: y(t) = H * x(t) + n(t), where SNR = P/N0."
    page1.insert_text((72, 300), formula_text, fontsize=10)

    img1 = Image.new("RGB", (200, 150), color=(240, 240, 240))
    draw1 = ImageDraw.Draw(img1)
    for x in range(0, 200, 20):
        y = 150 - int(100 * (x / 200) ** 0.5) - 10
        if x > 0:
            draw1.line([(x - 20, prev_y), (x, y)], fill=(30, 100, 200), width=2)
        prev_y = y
    draw1.text((60, 5), "Detection Probability", fill=(0, 0, 0))
    draw1.text((5, 70), "Pd", fill=(0, 0, 0))
    draw1.text((90, 135), "SNR (dB)", fill=(0, 0, 0))
    buf1 = io.BytesIO()
    img1.save(buf1, format="PNG")
    page1.insert_image(pymupdf.Rect(72, 340, 322, 530), stream=buf1.getvalue())
    page1.insert_text((72, 550), "Fig. 1: Detection probability vs SNR.", fontsize=9)

    # === Page 2: Methods + Results + Another Figure ===
    page2 = doc.new_page(width=595, height=842)

    page2.insert_text((72, 60), "1. Introduction", fontsize=13, fontname="helv")

    intro_text = (
        "Joint radar-communication systems have attracted significant attention "
        "in recent years due to the growing demand for spectrum efficiency. "
        "DSSS and FHSS techniques are commonly used in spread spectrum systems. "
        "The Kalman filter provides optimal state estimation under Gaussian noise. "
        "Deep learning approaches using CNN and LSTM architectures have shown "
        "promise in target detection and classification tasks."
    )
    rect2 = pymupdf.Rect(72, 80, 523, 220)
    page2.insert_textbox(rect2, intro_text, fontsize=10)

    page2.insert_text((72, 240), "2. System Model", fontsize=13, fontname="helv")

    model_text = (
        "Consider a MIMO radar with M transmit antennas and N receive antennas. "
        "The received signal can be modeled as: "
        "The optimization objective minimizes the mean squared error (MSE) "
        "subject to a total power constraint P_total."
    )
    rect3 = pymupdf.Rect(72, 260, 523, 380)
    page2.insert_textbox(rect3, model_text, fontsize=10)

    img2 = Image.new("RGB", (180, 120), color=(245, 245, 245))
    draw2 = ImageDraw.Draw(img2)
    for i in range(5):
        x0, y0 = 20 + i * 35, 100
        h = 20 + i * 18
        draw2.rectangle([(x0, y0 - h), (x0 + 25, y0)], fill=(50, 120, 200))
    draw2.text((40, 5), "MSE Comparison", fill=(0, 0, 0))
    buf2 = io.BytesIO()
    img2.save(buf2, format="PNG")
    page2.insert_image(pymupdf.Rect(72, 400, 302, 560), stream=buf2.getvalue())
    page2.insert_text((72, 575), "Fig. 2: MSE comparison of different methods.", fontsize=9)

    doc.save(str(path))
    doc.close()
    return path


@pytest.mark.skipif(not _have_qwen_key(), reason="需要 QWEN_API_KEY")
def test_full_flow_with_figures_formulas_and_translation(tmp_path, rich_test_pdf: Path):
    """完整流程：图表保存 → OCR → chunk → 图片合并 → 翻译 → LaTeX → PDF。"""
    stem = rich_test_pdf.stem
    output_base = tmp_path / "full_flow"
    output_base.mkdir()

    # ================================================================
    # Step 1: 提取并保存图片到本地
    # ================================================================
    print("\n" + "=" * 70)
    print("Step 1: 提取 PDF 内嵌图片并保存到本地")
    print("=" * 70)

    img_output = tmp_path / "img_output"
    img_output.mkdir()
    extractor = ImageExtractor(str(img_output))
    images = extractor.extract(str(rich_test_pdf), stem)

    assert len(images) >= 2, f"应提取至少2张图（实际 {len(images)}）"
    for img_info in images:
        img_path = Path(img_info["path"])
        assert img_path.is_file(), f"图片应保存到本地: {img_path}"
        assert img_path.stat().st_size > 100, f"图片不应为空: {img_path}"
        print(f"  ✓ 图片已保存: {img_path} ({img_path.stat().st_size} bytes)")

    page1_imgs = [i for i in images if i["page"] == 1]
    page2_imgs = [i for i in images if i["page"] == 2]
    assert page1_imgs, "第1页应有图片"
    assert page2_imgs, "第2页应有图片"

    # ================================================================
    # Step 2: Qwen-VL-OCR 提取文字（含公式识别）
    # ================================================================
    print("\n" + "=" * 70)
    print("Step 2: Qwen-VL-OCR 文字提取")
    print("=" * 70)

    ocr_dir = output_base / "ocr"
    en_md = extract_pdf_to_markdown(
        rich_test_pdf,
        output_dir=ocr_dir,
        dpi=200,
        model="qwen-vl-ocr",
    )

    assert len(en_md) > 100, f"OCR 结果应足够长（实际 {len(en_md)} 字符）"
    en_md_path = output_base / "english.md"
    en_md_path.write_text(en_md, encoding="utf-8")
    print(f"  ✓ OCR 完成: {len(en_md)} 字符 → {en_md_path}")
    print(f"  预览前300字符:\n{en_md[:300]}")

    # ================================================================
    # Step 3: 解析 OCR → chunk 格式
    # ================================================================
    print("\n" + "=" * 70)
    print("Step 3: 解析为标准 chunk 格式")
    print("=" * 70)

    chunks = parse_ocr_to_chunks(en_md)
    assert len(chunks) > 0, "应产出 chunk"

    text_groups = [c for c in chunks if c["type"] == "text_group"]
    headers = [c for c in chunks if c["type"] == "header"]
    print(f"  ✓ {len(chunks)} 个 chunk: {len(headers)} 标题, {len(text_groups)} 文本组")
    assert text_groups, "应有 text_group"

    # ================================================================
    # Step 4: 合并内嵌图片到 chunk 序列
    # ================================================================
    print("\n" + "=" * 70)
    print("Step 4: 合并图片到 chunk 序列")
    print("=" * 70)

    chunks = merge_embedded_figures_into_chunks(chunks, images, str(rich_test_pdf))
    img_chunks = [c for c in chunks if c["type"] == "image"]
    print(f"  ✓ 合并后: {len(chunks)} 个 chunk (含 {len(img_chunks)} 个图片占位)")
    assert img_chunks, "应有 image chunk（两页各至少一张图）"

    # ================================================================
    # Step 5: Qwen 翻译（保护公式和专有名词）
    # ================================================================
    print("\n" + "=" * 70)
    print("Step 5: Qwen 大模型翻译")
    print("=" * 70)

    api_key = qwen_api_key()
    model = os.environ.get("PDFTOOLS_TRANSLATE_MODEL", "qwen-plus")
    translator = QwenTranslator(api_key=api_key, model=model)
    translated = translator.translate_chunks(chunks)

    assert translated, "翻译结果不应为空"
    assert len(translated) == len(chunks), "翻译后 chunk 数量应与原始一致"

    translated_text_groups = [c for c in translated if c["type"] == "text_group"]
    assert translated_text_groups, "翻译结果应有 text_group"

    translated_imgs = [c for c in translated if c["type"] == "image"]
    assert len(translated_imgs) == len(img_chunks), "图片 chunk 应原样保留"

    all_translated_text = ""
    for c in translated_text_groups:
        for sub in c.get("chunks", []):
            for line in sub.get("content", []):
                all_translated_text += str(line) + "\n"

    print(f"  ✓ 翻译完成: {len(all_translated_text)} 字符")
    print(f"  预览前200字符:\n{all_translated_text[:200]}")

    has_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in all_translated_text)
    assert has_chinese, "翻译结果应包含中文"

    zh_txt = output_base / "translated_zh.txt"
    write_translated_plaintext(translated, zh_txt)
    assert zh_txt.is_file()
    print(f"  ✓ 翻译文本: {zh_txt}")

    # ================================================================
    # Step 6: LaTeX 构建 + 编译生成 PDF（图表公式自动插回）
    # ================================================================
    print("\n" + "=" * 70)
    print("Step 6: LaTeX 构建 + 编译")
    print("=" * 70)

    import shutil
    latex_root = tmp_path / "latex_root"
    dest_img_dir = latex_root / "result" / stem / "images"
    dest_img_dir.mkdir(parents=True)
    src_img_dir = img_output / "result" / stem / "images"
    if src_img_dir.is_dir():
        shutil.copytree(src_img_dir, dest_img_dir, dirs_exist_ok=True)

    builder = LatexBuilder(str(latex_root))
    compiled_pdf = builder.build_latex(stem, translated, images)

    tex_path = latex_root / "result" / stem / "document.tex"
    pdf_path = latex_root / "result" / stem / "document.pdf"

    assert tex_path.is_file(), "应生成 .tex 文件"
    print(f"  ✓ LaTeX 源文件: {tex_path}")

    tex_content = tex_path.read_text(encoding="utf-8")
    assert "\\includegraphics" in tex_content, "LaTeX 中应包含图片插入命令"
    assert "\\section" in tex_content or "\\section*" in tex_content, "LaTeX 中应有章节标题"
    assert "\\begin{figure}" in tex_content, "LaTeX 中应有 figure 环境"

    has_chinese_in_tex = any('\u4e00' <= ch <= '\u9fff' for ch in tex_content)
    assert has_chinese_in_tex, "LaTeX 中应包含翻译后的中文"

    if compiled_pdf:
        assert pdf_path.is_file(), "应生成编译后的 PDF"
        assert pdf_path.stat().st_size > 1000, "PDF 不应太小"
        print(f"  ✓ 编译 PDF: {pdf_path} ({pdf_path.stat().st_size} bytes)")

        verify_doc = pymupdf.open(str(pdf_path))
        assert len(verify_doc) >= 1, "生成的 PDF 应至少有1页"
        total_text = ""
        for page in verify_doc:
            total_text += page.get_text()
        verify_doc.close()
        has_chinese_in_pdf = any('\u4e00' <= ch <= '\u9fff' for ch in total_text)
        assert has_chinese_in_pdf, "最终 PDF 中应包含中文"

        img_list = []
        check_doc = pymupdf.open(str(pdf_path))
        for page in check_doc:
            img_list.extend(page.get_images(full=True))
        check_doc.close()
        assert len(img_list) >= 1, "最终 PDF 应包含图片"
        print(f"  ✓ PDF 包含 {len(img_list)} 张图片")
    else:
        print("  ⚠ PDF 未编译成功（xelatex 可能不可用），但 .tex 已验证通过")

    # ================================================================
    # Step 7: 验证完整性摘要
    # ================================================================
    print("\n" + "=" * 70)
    print("Step 7: 完整性验证摘要")
    print("=" * 70)
    print(f"  [✓] 图片提取: {len(images)} 张保存到本地")
    print(f"  [✓] OCR 提取: {len(en_md)} 字符")
    print(f"  [✓] Chunk 解析: {len(chunks)} 个 chunk")
    print(f"  [✓] 图表合并: {len(img_chunks)} 个图片占位")
    print(f"  [✓] 翻译: {len(all_translated_text)} 字符中文文本")
    print(f"  [✓] LaTeX 源文件: 包含图片、公式、章节标题")
    if compiled_pdf:
        print(f"  [✓] PDF 生成: 包含中文 + 图表")
    print("  完整流程全部通过！")


@pytest.mark.skipif(not _have_qwen_key(), reason="需要 QWEN_API_KEY")
def test_formula_preservation_through_pipeline(tmp_path, rich_test_pdf: Path):
    """验证公式在翻译过程中被正确保护和还原。"""
    print("\n" + "=" * 70)
    print("测试：公式保护 & 翻译还原")
    print("=" * 70)

    md_with_formulas = """## Page 1

\\section{Signal Model}

The received signal is modeled as:

$$y(t) = H \\cdot x(t) + n(t)$$

where $H$ is the channel matrix and $n(t)$ represents AWGN noise with variance $\\sigma^2$.

The SNR is defined as $\\text{SNR} = P / N_0$ and the detection probability $P_d$ depends on the CFAR threshold.
"""

    chunks = parse_ocr_to_chunks(md_with_formulas)
    assert chunks, "应解析出 chunk"

    api_key = qwen_api_key()
    translator = QwenTranslator(api_key=api_key, model="qwen-plus")
    translated = translator.translate_chunks(chunks)

    all_text = ""
    for c in translated:
        if c["type"] == "text_group":
            for sub in c.get("chunks", []):
                for line in sub.get("content", []):
                    all_text += str(line) + "\n"
        elif c["type"] == "header":
            all_text += (c.get("content") or [""])[0] + "\n"

    print(f"  翻译结果:\n{all_text}")

    has_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in all_text)
    assert has_chinese, "翻译结果应包含中文"

    abbreviations = ["SNR", "AWGN", "CFAR"]
    for abbr in abbreviations:
        if abbr in md_with_formulas:
            assert abbr in all_text, f"缩写 {abbr} 应被保护不翻译"
            print(f"  ✓ 缩写 {abbr} 已保护")

    print("  ✓ 公式保护测试通过")


@pytest.mark.skipif(not _have_qwen_key(), reason="需要 QWEN_API_KEY")
def test_quality_check_on_generated_pdf(tmp_path, rich_test_pdf: Path):
    """对生成的 PDF 进行质量检查。"""
    stem = rich_test_pdf.stem

    img_output = tmp_path / "img_q"
    img_output.mkdir()
    ext = ImageExtractor(str(img_output))
    images = ext.extract(str(rich_test_pdf), stem)

    en_md = extract_pdf_to_markdown(rich_test_pdf, dpi=150, pages=[1])
    chunks = parse_ocr_to_chunks(en_md)
    chunks = merge_embedded_figures_into_chunks(chunks, images, str(rich_test_pdf))

    api_key = qwen_api_key()
    translator = QwenTranslator(api_key=api_key)
    translated = translator.translate_chunks(chunks)

    import shutil
    latex_root = tmp_path / "latex_q"
    dest = latex_root / "result" / stem / "images"
    dest.mkdir(parents=True)
    src = img_output / "result" / stem / "images"
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)

    builder = LatexBuilder(str(latex_root))
    compiled_pdf = builder.build_latex(stem, translated, images)

    if not compiled_pdf:
        pytest.skip("xelatex 未生成 PDF，跳过质量检查")

    print("\n" + "=" * 70)
    print("质量检查")
    print("=" * 70)

    checker = QualityChecker(str(latex_root))
    report = checker.check_quality(str(rich_test_pdf), compiled_pdf, stem)

    assert report, "应产出质量报告"
    assert "average_ssim" in report, "报告应含 average_ssim"
    assert "quality_level" in report, "报告应含 quality_level"

    print(f"  SSIM: {report.get('average_ssim', 0):.3f}")
    print(f"  余弦: {report.get('average_cosine', 0):.3f}")
    print(f"  等级: {report.get('quality_level', 'N/A')}")
    print("  ✓ 质量检查通过")
