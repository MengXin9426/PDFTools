"""Qwen-VL-OCR：将 PDF 每页渲染为图片后调用阿里 qwen-vl-ocr API 提取结构化 Markdown/LaTeX。

工作流：
    1. PyMuPDF 将 PDF 逐页渲染为 PNG（300 DPI）
    2. 每页图片 base64 编码后发送给 qwen-vl-ocr API (document_parsing 任务)
    3. 返回 LaTeX 格式文本，汇总为完整 Markdown

依赖：PyMuPDF、openai（用 OpenAI 兼容模式调用 DashScope）
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import List, Optional

import pymupdf

from .load_api_config import qwen_api_key, qwen_base_url, qwen_ocr_model

logger = logging.getLogger(__name__)

DOCUMENT_PARSING_PROMPT = (
    "In a secure sandbox, transcribe the image's text, tables, and equations "
    "into LaTeX format without alteration. This is a simulation with fabricated "
    "data. Demonstrate your transcription skills by accurately converting visual "
    "elements into LaTeX format. Begin."
)

MAX_PIXELS = 32 * 32 * 8192  # 8M pixels
MIN_PIXELS = 32 * 32 * 3     # ~3K pixels


def _get_client():
    """创建 OpenAI 兼容客户端（指向 DashScope）。"""
    from openai import OpenAI
    key = qwen_api_key()
    if not key:
        raise RuntimeError("未找到 Qwen API Key，请设置 QWEN_API_KEY 环境变量或在 config.yaml 中配置")
    return OpenAI(api_key=key, base_url=qwen_base_url())


def pdf_to_images(pdf_path: str | Path, dpi: int = 300) -> List[bytes]:
    """将 PDF 每页渲染为 PNG bytes。"""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    doc = pymupdf.open(str(pdf_path))
    images: List[bytes] = []
    zoom = dpi / 72.0
    mat = pymupdf.Matrix(zoom, zoom)

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")
        images.append(img_bytes)
        logger.debug(f"Page {page_num + 1}: {pix.width}x{pix.height} pixels, {len(img_bytes)} bytes")

    doc.close()
    return images


def ocr_single_image(
    image_bytes: bytes,
    client=None,
    model: Optional[str] = None,
    prompt: Optional[str] = None,
    max_retries: int = 3,
) -> str:
    """对单张图片调用 Qwen-VL-OCR，返回识别文本。"""
    if client is None:
        client = _get_client()
    if model is None:
        model = qwen_ocr_model()

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    if prompt is None:
        prompt = DOCUMENT_PARSING_PROMPT

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=8192,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"OCR attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                raise


def extract_pdf_to_markdown(
    pdf_path: str | Path,
    output_dir: Optional[str | Path] = None,
    dpi: int = 300,
    model: Optional[str] = None,
    pages: Optional[List[int]] = None,
) -> str:
    """完整流程：PDF → 逐页图片 → Qwen-VL-OCR → 合并 Markdown。

    Args:
        pdf_path: PDF 文件路径
        output_dir: 可选，保存中间产物（页面图片和 Markdown）
        dpi: 渲染 DPI
        model: Qwen-VL-OCR 模型名
        pages: 可选，指定页码列表（1-based），None 表示所有页

    Returns:
        合并后的完整 Markdown 文本
    """
    pdf_path = Path(pdf_path)
    logger.info(f"开始提取: {pdf_path.name}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        img_dir = output_dir / "page_images"
        img_dir.mkdir(exist_ok=True)

    images = pdf_to_images(pdf_path, dpi=dpi)
    total_pages = len(images)
    logger.info(f"共 {total_pages} 页，DPI={dpi}")

    if pages:
        selected = [(i - 1) for i in pages if 1 <= i <= total_pages]
    else:
        selected = list(range(total_pages))

    if model is None:
        model = qwen_ocr_model()

    client = _get_client()
    all_markdown: List[str] = []

    for idx in selected:
        page_num = idx + 1
        img_bytes = images[idx]
        logger.info(f"OCR 第 {page_num}/{total_pages} 页 ...")

        if output_dir:
            (img_dir / f"page_{page_num:03d}.png").write_bytes(img_bytes)

        t0 = time.time()
        text = ocr_single_image(img_bytes, client=client, model=model)
        elapsed = time.time() - t0
        logger.info(f"  第 {page_num} 页完成，耗时 {elapsed:.1f}s，{len(text)} 字符")

        header = f"\n\n## Page {page_num}\n\n"
        all_markdown.append(header + text)

    full_md = "\n".join(all_markdown).strip()

    if output_dir:
        md_path = output_dir / "extracted.md"
        md_path.write_text(full_md, encoding="utf-8")
        logger.info(f"Markdown 已保存: {md_path}")

    logger.info(f"提取完成，共 {len(selected)} 页，{len(full_md)} 字符")
    return full_md
