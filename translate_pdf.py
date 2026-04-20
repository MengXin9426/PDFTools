#!/usr/bin/env python3
"""PDF 学术论文翻译 Pipeline（完整版）

工作流：
    1. PDF → Qwen-VL-OCR → 结构化 Markdown/LaTeX（含公式、表格）
    2. OCR 结果 → 标准 chunk 格式（header / text_group / latex_raw / image）
    3. DocLayout-YOLO 版面检测 → 裁剪 figure/table 区域
    4. Qwen 翻译 chunk（保护公式/专有名词）
    5. LaTeX 编译生成中文 PDF
    6. 质量检查（原 PDF vs 翻译 PDF）

用法：
    python translate_pdf.py input/paper.pdf
    python translate_pdf.py input/paper.pdf --pages 1-3
    python translate_pdf.py input/paper.pdf --ocr-only
    python translate_pdf.py input/paper.pdf --from-ocr -o output/run1 --no-latex
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from skills.qwen_vl_ocr import extract_pdf_to_markdown
from skills.ocr_to_chunks import parse_ocr_to_chunks
from skills.translator.qwen_translator import QwenTranslator
from skills.latex_builder import LatexBuilder
from skills.latex_builder.plaintext import write_translated_plaintext
from skills.quality_checker import QualityChecker
from skills.load_api_config import (
    active_translate_api_key,
    translate_backend as cfg_translate_backend,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_pages(pages_str: str, total: int) -> Optional[List[int]]:
    """解析页码范围字符串，如 '1-3,5,8-10'。"""
    if not pages_str:
        return None
    result = []
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return sorted(set(p for p in result if 1 <= p <= total))


def main():
    parser = argparse.ArgumentParser(
        description="PDF 学术论文翻译（Qwen-VL-OCR + DocLayout-YOLO + Qwen 翻译 + LaTeX 编译）",
    )
    parser.add_argument("pdf", help="输入 PDF 文件路径")
    parser.add_argument("-o", "--output", help="输出目录（默认 output/<文件名>）")
    parser.add_argument("--pages", help="页码范围（如 1-3,5），默认全部页")
    parser.add_argument("--dpi", type=int, default=300, help="PDF 渲染 DPI（默认 300）")
    parser.add_argument("--ocr-model", default="qwen-vl-ocr", help="OCR 模型")
    parser.add_argument("--translate-model", default=None, help="翻译模型（默认读取 config.yaml）")
    parser.add_argument("--ocr-only", action="store_true", help="仅 OCR 提取，不翻译")
    parser.add_argument(
        "--from-ocr",
        action="store_true",
        help="跳过 OCR，直接读取输出目录中的 english.md",
    )
    parser.add_argument("--no-latex", action="store_true", help="跳过 LaTeX 编译")
    parser.add_argument("--no-quality", action="store_true", help="跳过质量检查")
    parser.add_argument("--no-layout-detect", action="store_true", help="跳过 DocLayout-YOLO 版面检测")
    parser.add_argument(
        "--backend",
        choices=["qwen", "vllm"],
        default=None,
        help="翻译后端（默认读取 config.yaml 的 translate_backend）",
    )
    parser.add_argument(
        "--layout",
        choices=["auto", "twocolumn", "onecolumn"],
        default=None,
        help="排版模式（默认读取 config.yaml 的 latex.layout）",
    )
    args = parser.parse_args()
    if args.from_ocr and args.ocr_only:
        parser.error("--from-ocr 与 --ocr-only 不能同时使用")

    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        print(f"错误：PDF 文件不存在: {pdf_path}")
        sys.exit(1)

    stem = pdf_path.stem
    output_dir = Path(args.output) if args.output else Path("output") / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = None
    if args.pages:
        import pymupdf
        doc = pymupdf.open(str(pdf_path))
        total = len(doc)
        doc.close()
        pages = parse_pages(args.pages, total)
        logger.info(f"指定页码: {pages}")

    t_total_start = time.time()
    t_ocr = 0.0
    en_md_path = output_dir / "english.md"

    # ======================================================
    # Step 1: Qwen-VL-OCR 提取
    # ======================================================
    if args.from_ocr:
        if not en_md_path.is_file():
            logger.error(f"--from-ocr 需要已有文件: {en_md_path}")
            sys.exit(1)
        logger.info("=" * 60)
        logger.info("Step 1: 跳过 OCR，读取已有 english.md")
        logger.info("=" * 60)
        en_md = en_md_path.read_text(encoding="utf-8")
        logger.info(f"已载入 {len(en_md)} 字符 ← {en_md_path}")
    else:
        logger.info("=" * 60)
        logger.info("Step 1: Qwen-VL-OCR 提取 PDF → Markdown/LaTeX")
        logger.info("=" * 60)
        t0 = time.time()

        en_md = extract_pdf_to_markdown(
            pdf_path,
            output_dir=output_dir / "ocr",
            dpi=args.dpi,
            model=args.ocr_model,
            pages=pages,
        )

        en_md_path.write_text(en_md, encoding="utf-8")
        t_ocr = time.time() - t0
        logger.info(f"OCR 完成，耗时 {t_ocr:.1f}s，{len(en_md)} 字符 → {en_md_path}")

    if args.ocr_only:
        logger.info("仅 OCR 模式，完成。")
        return

    # ======================================================
    # Step 2: 解析为标准 chunk 格式
    # ======================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 2: 解析 OCR 输出 → 标准 chunk 格式")
    logger.info("=" * 60)

    chunks = parse_ocr_to_chunks(en_md)

    from collections import Counter
    type_counts = Counter(c["type"] for c in chunks)
    logger.info(
        f"解析完成：{len(chunks)} 个 chunk"
        f"（{type_counts.get('header', 0)} 标题, "
        f"{type_counts.get('text_group', 0)} 文本组, "
        f"{type_counts.get('latex_raw', 0)} 公式/表格环境, "
        f"{type_counts.get('image', 0)} 图片）"
    )

    # ======================================================
    # Step 3: DocLayout-YOLO 版面检测 + 图表裁剪
    # ======================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("Step 3: 版面检测与图表提取")
    logger.info("=" * 60)

    page_img_dir = str(output_dir / "ocr" / "page_images")
    result_dir = Path("output") / "result" / stem
    result_dir.mkdir(parents=True, exist_ok=True)
    images_dir = str(result_dir / "images")

    detected_images: List[dict] = []

    if not args.no_layout_detect:
        try:
            from skills.layout_detector import LayoutDetector
            t_ld = time.time()
            detector = LayoutDetector()
            detected_images = detector.detect_and_crop_figures(
                page_img_dir, images_dir, conf=0.3,
            )
            logger.info(f"版面检测完成，耗时 {time.time() - t_ld:.1f}s，"
                        f"裁剪 {len(detected_images)} 张图表区域")
        except Exception as e:
            logger.warning(f"DocLayout-YOLO 版面检测失败: {e}")
            logger.warning("将使用 page_images 作为 fallback")
    else:
        logger.info("跳过版面检测（--no-layout-detect）")

    # ======================================================
    # Step 4: Qwen 翻译
    # ======================================================
    logger.info("")
    backend = args.backend or cfg_translate_backend()
    logger.info("=" * 60)
    logger.info(f"Step 4: 翻译 chunk → 中文（后端: {backend}）")
    logger.info("=" * 60)
    t1 = time.time()

    api_key = active_translate_api_key()
    translator = QwenTranslator(api_key=api_key, model=args.translate_model, backend=backend)
    translated_chunks = translator.translate_chunks(chunks)

    t_trans = time.time() - t1
    logger.info(f"翻译完成，耗时 {t_trans:.1f}s")

    zh_txt_path = output_dir / "translated_zh.txt"
    write_translated_plaintext(translated_chunks, str(zh_txt_path))
    logger.info(f"中文纯文本 → {zh_txt_path}")

    zh_md_path = output_dir / "chinese.md"
    _write_translated_markdown(translated_chunks, zh_md_path)
    logger.info(f"中文 Markdown → {zh_md_path}")

    # ======================================================
    # Step 5: LaTeX 编译
    # ======================================================
    compiled_pdf = None
    if not args.no_latex:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Step 5: LaTeX 编译生成中文 PDF")
        logger.info("=" * 60)
        t2 = time.time()

        builder = LatexBuilder("output", layout=args.layout)
        compiled_pdf = builder.build_latex(
            stem, translated_chunks, detected_images,
            page_images_dir=page_img_dir,
        )

        t_latex = time.time() - t2
        inner_tex = result_dir / "document.tex"
        if inner_tex.is_file():
            shutil.copy2(inner_tex, output_dir / "document.tex")
            logger.info(f"已复制 LaTeX 源到: {output_dir / 'document.tex'}")
        if compiled_pdf:
            logger.info(f"LaTeX 编译成功，耗时 {t_latex:.1f}s → {compiled_pdf}")
        else:
            logger.warning(f"LaTeX 编译失败（耗时 {t_latex:.1f}s），.tex 文件已保存")

    # ======================================================
    # Step 6: 质量检查
    # ======================================================
    if not args.no_quality and compiled_pdf:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Step 6: 质量检查")
        logger.info("=" * 60)
        t3 = time.time()

        checker = QualityChecker("output")
        report = checker.check_quality(str(pdf_path), compiled_pdf, stem)

        t_quality = time.time() - t3
        logger.info(f"质量检查完成，耗时 {t_quality:.1f}s")
        if report:
            avg_ssim = report.get("average_ssim", "N/A")
            logger.info(f"  平均 SSIM: {avg_ssim}")

    # ======================================================
    # 总结
    # ======================================================
    t_total = time.time() - t_total_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline 完成！")
    logger.info(f"  OCR 耗时:     {t_ocr:.1f}s")
    logger.info(f"  翻译耗时:     {t_trans:.1f}s")
    logger.info(f"  总耗时:       {t_total:.1f}s")
    logger.info(f"  英文 Markdown: {en_md_path}")
    logger.info(f"  中文纯文本:    {zh_txt_path}")
    logger.info(f"  中文 Markdown: {zh_md_path}")
    if compiled_pdf:
        logger.info(f"  翻译 PDF:     {compiled_pdf}")
    logger.info("=" * 60)


def _write_translated_markdown(chunks: List[dict], path: Path):
    """将翻译后的 chunks 写为可读 Markdown。"""
    lines: List[str] = []
    for chunk in chunks:
        ctype = chunk.get("type", "")
        if ctype == "header":
            content = chunk.get("content", [])
            title = content[0] if content else ""
            lines.append(f"\n## {title}\n")
        elif ctype == "text_group":
            for sub in chunk.get("chunks", []):
                for line in sub.get("content", []):
                    lines.append(line)
            lines.append("")
        elif ctype == "latex_raw":
            lines.append("")
            lines.append(chunk.get("content", ""))
            lines.append("")
        elif ctype == "image":
            page = chunk.get("page", "?")
            lines.append(f"\n[图片 - 第{page}页]\n")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
