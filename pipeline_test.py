#!/usr/bin/env python3
"""
PDF 翻译工作流：文本提取 → 切块 → Qwen 大模型翻译 → LaTeX → 编译 PDF。

提取模式：
  - 默认：PyMuPDF 直接提取（可加 ``--latex-ocr``）
  - ``--extract-kit``：调用 PDF-Extract-Kit 深度解析（版面 + 公式 + OCR）

翻译 ``--translator``：
  - ``auto``（默认）：若已配置 Qwen 密钥则用 Qwen，否则 Google
  - ``qwen``：强制 Qwen 大模型
  - ``google``：Google 翻译
  - ``mock``：仅离线占位（测试用）

输出：``output/result/<PDF名>/`` 下 ``translated_zh.txt``、``document.tex``、``document.pdf``。
环境变量 ``PDFTOOLS_MAX_PAGES=N`` 可限制处理页数。
"""

import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from skills.load_api_config import qwen_api_key
from skills.pdf_searcher import PDFSearcher
from skills.text_extractor import TextExtractor
from skills.image_extractor import ImageExtractor
from skills.text_chunker import TextChunker
from skills.translator.mock_translator import MockTranslator
from skills.translator import Translator, QwenTranslator
from skills.latex_builder import LatexBuilder, write_translated_plaintext
from skills.quality_checker import QualityChecker


def _build_translator(mode: str):
    mode = (mode or "auto").lower()
    if mode == "mock":
        return MockTranslator(), "mock"
    if mode == "google":
        return Translator(source_lang="en", target_lang="zh-CN"), "google"
    if mode == "qwen":
        key = qwen_api_key()
        if not key:
            raise SystemExit(
                "未配置 Qwen API 密钥：请设置环境变量 QWEN_API_KEY，"
                "或在项目根目录 config.yaml 中填写 api.qwen.api_key"
            )
        return QwenTranslator(api_key=key), "qwen"
    if mode == "auto":
        key = qwen_api_key()
        if key:
            return QwenTranslator(api_key=key), "qwen(auto)"
        return Translator(source_lang="en", target_lang="zh-CN"), "google(auto)"
    raise SystemExit(f"未知 --translator: {mode}")


def _print_output_summary(
    pdf_name: str,
    output_dir: str,
    generated_pdf: Optional[str],
    explicit_translated: Optional[str] = None,
) -> dict:
    root = Path(output_dir).resolve() / "result" / pdf_name
    out_root = Path(output_dir).resolve()
    paths = {
        "result_dir": str(root),
        "translated_txt": str((root / "translated_zh.txt").resolve()),
        "document_tex": str((root / "document.tex").resolve()),
        "document_pdf": str((root / "document.pdf").resolve()) if generated_pdf else None,
        "translated_pdf_fixed": explicit_translated
        or str((out_root / f"{pdf_name}_translated.pdf").resolve()),
        "extracted_txt": str((root / "text" / "extracted.txt").resolve()),
    }
    print("\n" + "=" * 60)
    print("输出文件（绝对路径）")
    print("=" * 60)
    print(f"  目录:     {paths['result_dir']}")
    print(f"  翻译文本: {paths['translated_txt']}")
    print(f"  LaTeX:    {paths['document_tex']}")
    if paths["document_pdf"] and Path(paths["document_pdf"]).exists():
        print(f"  PDF:      {paths['document_pdf']}")
    else:
        print("  PDF:      （未生成 — 请安装 texlive-xetex 或 tectonic 后重试）")
    if explicit_translated and Path(explicit_translated).exists():
        print(f"  翻译PDF(固定名): {explicit_translated}")
    elif Path(paths["translated_pdf_fixed"]).exists():
        print(f"  翻译PDF(固定名): {paths['translated_pdf_fixed']}")
    print("=" * 60)
    return paths


class PDFTranslatePipeline:
    """PDF 处理工作流"""

    def __init__(
        self,
        input_dir: str = "input",
        output_dir: str = "output",
        translator_mode: str = "auto",
        use_latex_ocr: bool = False,
        use_extract_kit: bool = False,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.use_extract_kit = use_extract_kit
        self.translator, self._translator_label = _build_translator(translator_mode)

        self.searcher = PDFSearcher(input_dir)
        self.text_extractor = TextExtractor(output_dir, use_latex_ocr=use_latex_ocr)
        self.image_extractor = ImageExtractor(output_dir)
        self.text_chunker = TextChunker()
        self.latex_builder = LatexBuilder(output_dir)
        self.quality_checker = QualityChecker(output_dir)

    def run(self, pdf_name: Optional[str] = None) -> dict:
        extract_label = "PDF-Extract-Kit" if self.use_extract_kit else "PyMuPDF"
        print("=" * 60)
        print("PDF 翻译工作流")
        print(f"提取后端: {extract_label}")
        print(f"翻译后端: {self._translator_label}")
        if not self.use_extract_kit:
            print(f"LaTeX-OCR: {'开启' if self.text_extractor.use_latex_ocr else '关闭'}")
        print("=" * 60)

        results = {}

        print("\n[1/7] 搜索 PDF 文件...")
        pdf_files = self.searcher.search_pdfs()

        if not pdf_files:
            print("未找到任何 PDF 文件")
            return {"success": False, "error": "No PDF files found"}

        if pdf_name:
            pdf_files = [f for f in pdf_files if f["name"] == pdf_name]
            if not pdf_files:
                print(f"未找到指定的 PDF 文件: {pdf_name}")
                return {"success": False, "error": f"PDF not found: {pdf_name}"}

        for pdf_info in pdf_files:
            pdf_path = pdf_info["path"]
            pdf_name = pdf_info["name"]

            print(f"\n处理 PDF: {pdf_name} ({pdf_info['pages']} 页)")
            print("-" * 60)

            try:
                max_pages = int(os.environ.get("PDFTOOLS_MAX_PAGES", "0"))
                if max_pages > 0:
                    print(f"\n  PDFTOOLS_MAX_PAGES={max_pages}，仅处理前 {max_pages} 页")

                if self.use_extract_kit:
                    from skills.pdf_extract_kit_bridge import run_extract_kit

                    print("\n[2/7] 提取文本（PDF-Extract-Kit）...")
                    extracted_text, images = run_extract_kit(
                        pdf_path, pdf_name, self.output_dir,
                        max_pages=max_pages,
                    )
                    results[pdf_name] = {"extracted": extracted_text}
                    print(f"\n[3/7] 图片已由 Kit 裁剪（{len(images)} 张）")
                    results[pdf_name]["images"] = images
                else:
                    print("\n[2/7] 提取文本...")
                    extracted_text = self.text_extractor.extract(pdf_path, pdf_name)
                    if max_pages > 0:
                        extracted_text["pages"] = extracted_text["pages"][:max_pages]
                        extracted_text["total_pages"] = len(extracted_text["pages"])
                    results[pdf_name] = {"extracted": extracted_text}

                    print("\n[3/7] 提取图片...")
                    images = self.image_extractor.extract(pdf_path, pdf_name)
                    if max_pages > 0:
                        images = [im for im in images if im.get("page", 0) <= max_pages]
                    results[pdf_name]["images"] = images

                print("\n[4/7] 文本分块...")
                chunks = self.text_chunker.chunk(extracted_text)
                merged_chunks = self.text_chunker.merge_for_translation(chunks)
                results[pdf_name]["chunks"] = merged_chunks

                print("\n[5/7] 翻译...")
                translated_chunks = self.translator.translate_chunks(merged_chunks)
                results[pdf_name]["translated"] = translated_chunks

                result_root = Path(self.output_dir) / "result" / pdf_name
                txt_out = result_root / "translated_zh.txt"
                write_translated_plaintext(translated_chunks, txt_out)
                print(f"  -> 已写入翻译文本: {txt_out.resolve()}")

                print("\n[6/7] 构建 LaTeX 文档...")
                generated_pdf = self.latex_builder.build_latex(pdf_name, translated_chunks, images)

                exp = (
                    str(self.latex_builder.explicit_translated_pdf)
                    if self.latex_builder.explicit_translated_pdf
                    else None
                )
                paths = _print_output_summary(
                    pdf_name, self.output_dir, generated_pdf, exp
                )
                results[pdf_name]["output_paths"] = paths
                if exp:
                    results[pdf_name]["translated_pdf_fixed"] = exp

                if generated_pdf:
                    results[pdf_name]["generated_pdf"] = generated_pdf
                    print("\n[7/7] 质量检查...")
                    quality_report = self.quality_checker.check_quality(
                        pdf_path, generated_pdf, pdf_name
                    )
                    results[pdf_name]["quality"] = quality_report
                    print(f"\n{pdf_name} 处理完成")
                    print(f"  质量等级: {quality_report.get('quality_level', 'Unknown')}")
                    print(f"  平均SSIM: {quality_report.get('average_ssim', 0):.3f}")
                else:
                    tex_path = result_root / "document.tex"
                    print(f"\n{pdf_name} 未编译出 PDF")
                    print("\n[7/7] 质量检查...")
                    quality_report = self.quality_checker.check_quality(
                        pdf_path, str(tex_path), pdf_name
                    )
                    results[pdf_name]["quality"] = quality_report

            except Exception as e:
                print(f"\n处理 {pdf_name} 时出错: {e}")
                results[pdf_name] = {"error": str(e)}
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 60)
        print("工作流执行完成")
        print("=" * 60)

        return {"success": True, "results": results}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="PDF 翻译：默认 Qwen 大模型；可选 LaTeX-OCR 或 PDF-Extract-Kit"
    )
    parser.add_argument("--pdf", "-p", help="指定处理的 PDF 文件名（不含扩展名）")
    parser.add_argument(
        "--translator", "-t",
        choices=["auto", "qwen", "google", "mock"],
        default="auto",
        help="auto=有 Qwen 密钥则大模型否则 Google；qwen=强制大模型；google；mock=离线占位",
    )

    extract_group = parser.add_mutually_exclusive_group()
    extract_group.add_argument(
        "--latex-ocr", action="store_true",
        help="启用 LaTeX-OCR 对小图块做公式识别",
    )
    extract_group.add_argument(
        "--extract-kit", action="store_true",
        help="使用 PDF-Extract-Kit 做深度版面/公式/OCR 解析",
    )

    args = parser.parse_args()

    pipeline = PDFTranslatePipeline(
        input_dir="input",
        output_dir="output",
        translator_mode=args.translator,
        use_latex_ocr=args.latex_ocr,
        use_extract_kit=args.extract_kit,
    )
    result = pipeline.run(args.pdf)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
