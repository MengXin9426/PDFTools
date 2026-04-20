"""
PDF文本提取器
支持直接提取和OCR两种方式
保留文档结构信息

可选：结合本地 LaTeX-OCR（pix2tex）对小面积图块做公式识别，在图块前插入 ``$$...$$`` 文本行。
"""

import io
import os
from pathlib import Path
from typing import Dict, List

import pymupdf


class TextExtractor:
    """PDF文本提取器"""

    def __init__(self, output_dir: str = "output", use_latex_ocr: bool = False):
        """
        初始化提取器

        Args:
            output_dir: 输出文件夹
            use_latex_ocr: 是否用 LaTeX-OCR 增强小块图像中的公式（需本机可 import pix2tex）
        """
        self.output_dir = Path(output_dir)
        self.use_latex_ocr = use_latex_ocr
        self._latex_ocr_max = int(os.environ.get("PDFTOOLS_MAX_LATEX_OCR", "120"))

    def extract(self, pdf_path: str, pdf_name: str) -> Dict:
        """
        提取PDF文本内容

        Args:
            pdf_path: PDF文件路径
            pdf_name: PDF文件名（不含扩展名）

        Returns:
            提取结果字典
        """
        print(f"\n提取文本: {pdf_name}")

        # 创建输出目录
        text_dir = self.output_dir / "result" / pdf_name / "text"
        text_dir.mkdir(parents=True, exist_ok=True)

        # 尝试直接提取
        result = self._extract_direct(pdf_path)

        # 检查提取质量
        if self._is_low_quality(result):
            print("  ⚠ 直接提取质量较差，尝试OCR...")
            result = self._extract_ocr(pdf_path)

        if self.use_latex_ocr and result.get("method") == "direct":
            self._inject_latex_ocr_blocks(pdf_path, result)
            result["method"] = "direct+latex_ocr"

        # 保存结构化文本
        output_file = text_dir / "extracted.txt"
        self._save_structured_text(result, output_file)

        print(f"  ✓ 文本已保存到: {output_file}")
        return result

    def _extract_direct(self, pdf_path: str) -> Dict:
        """直接提取文本"""
        doc = pymupdf.open(pdf_path)
        pages_data = []

        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")
            page_data = {
                'page': page_num + 1,
                'blocks': self._parse_blocks(blocks["blocks"]),
                'has_text': len(blocks["blocks"]) > 0
            }
            pages_data.append(page_data)

        doc.close()

        return {
            'method': 'direct',
            'pages': pages_data,
            'total_pages': len(pages_data)
        }

    def _parse_blocks(self, blocks: List) -> List[Dict]:
        """解析文本块"""
        parsed = []

        for block in blocks:
            if block["type"] == 0:  # 文本块
                parsed.append({
                    'type': 'text',
                    'bbox': block["bbox"],
                    'lines': self._parse_lines(block.get("lines", []))
                })
            elif block["type"] == 1:  # 图片块
                parsed.append({
                    'type': 'image',
                    'bbox': block["bbox"]
                })

        return parsed

    def _inject_latex_ocr_blocks(self, pdf_path: str, result: Dict) -> None:
        """对版面中面积较小的 image 块做公式识别，并在其前插入文本块（保留原 image 占位）。"""
        try:
            from ..latex_ocr_bridge import LatexOcrBridge
        except ImportError:
            print("  ⚠ LaTeX-OCR 桥接模块不可用，跳过公式增强")
            return

        bridge = LatexOcrBridge.instance()
        if not bridge.is_available():
            print(f"  ⚠ LaTeX-OCR 不可用: {bridge.last_error() or '路径无效'}")
            return

        from PIL import Image

        used = 0
        doc = pymupdf.open(pdf_path)
        try:
            for page_data in result.get("pages", []):
                page = doc[page_data["page"] - 1]
                page_area = max(page.rect.width * page.rect.height, 1.0)
                new_blocks: List[Dict] = []

                for block in page_data.get("blocks", []):
                    if block.get("type") != "image":
                        new_blocks.append(block)
                        continue

                    bbox = block.get("bbox")
                    if not bbox or len(bbox) < 4:
                        new_blocks.append(block)
                        continue

                    rect = pymupdf.Rect(bbox)
                    w, h = rect.width, rect.height
                    if w < 10 or h < 10:
                        new_blocks.append(block)
                        continue

                    area_ratio = (w * h) / page_area
                    # 整页/大图多为插图，不做公式 OCR
                    if area_ratio > 0.12 or w > 520 or h > 520:
                        new_blocks.append(block)
                        continue

                    if used >= self._latex_ocr_max:
                        new_blocks.append(block)
                        continue

                    try:
                        pix = page.get_pixmap(clip=rect, dpi=200)
                        pil = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                    except Exception as e:  # noqa: BLE001
                        print(f"  ⚠ LaTeX-OCR 裁图失败: {e}")
                        new_blocks.append(block)
                        continue

                    latex = bridge.predict_pil(pil)
                    used += 1
                    if latex:
                        new_blocks.append(
                            {
                                "type": "text",
                                "bbox": list(bbox),
                                "lines": [f"$$ {latex} $$"],
                                "source": "latex_ocr",
                            }
                        )
                    new_blocks.append(block)

                page_data["blocks"] = new_blocks
        finally:
            doc.close()

        print(f"  ✓ LaTeX-OCR 已处理 {used} 个小图块（上限 {self._latex_ocr_max}）")

    def _parse_lines(self, lines: List) -> List[str]:
        """解析文本行"""
        text_lines = []
        for line in lines:
            spans = line.get("spans", [])
            line_text = "".join(span["text"] for span in spans)
            if line_text.strip():
                text_lines.append(line_text)
        return text_lines

    def _is_low_quality(self, result: Dict) -> bool:
        """检查提取质量"""
        text_count = sum(
            1 for p in result['pages']
            for b in p['blocks']
            if b['type'] == 'text'
        )
        return text_count < result['total_pages'] * 2

    def _extract_ocr(self, pdf_path: str) -> Dict:
        """使用OCR提取文本"""
        try:
            import pytesseract
            from PIL import Image
            import io

            doc = pymupdf.open(pdf_path)
            pages_data = []

            for page_num, page in enumerate(doc):
                # 渲染页面为图片
                pix = page.get_pixmap(dpi=300)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))

                # OCR识别
                text = pytesseract.image_to_string(img)

                lines = [ln for ln in (text or "").splitlines() if ln.strip()]
                page_data = {
                    'page': page_num + 1,
                    'ocr_text': text,
                    'blocks': [{'type': 'text', 'lines': lines or ([text] if text else [])}],
                }
                pages_data.append(page_data)

            doc.close()

            return {
                'method': 'ocr',
                'pages': pages_data,
                'total_pages': len(pages_data)
            }

        except Exception as e:
            print(f"  ⚠ OCR失败: {e}，使用直接提取结果")
            return self._extract_direct(pdf_path)

    def _save_structured_text(self, result: Dict, output_path: Path):
        """保存结构化文本"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# 提取方法: {result['method']}\n")
            f.write(f"# 总页数: {result['total_pages']}\n\n")

            for page_data in result['pages']:
                f.write(f"## 第 {page_data['page']} 页\n\n")

                for block in page_data.get('blocks', []):
                    if block['type'] == 'text':
                        if 'lines' in block:
                            for line in block['lines']:
                                f.write(f"{line}\n")
                        elif 'text' in block:
                            f.write(f"{block['text']}\n")
                        f.write("\n")
