"""
高级图片提取器
使用VLA和深度学习方法提取论文插图
"""

import os
import io
from pathlib import Path
from typing import List, Dict, Tuple
import pymupdf
from PIL import Image
import cv2
import numpy as np


class AdvancedImageExtractor:
    """高级图片提取器"""

    def __init__(self, output_dir: str = "output"):
        """
        初始化提取器

        Args:
            output_dir: 输出文件夹
        """
        self.output_dir = Path(output_dir)

    def extract_with_classification(self, pdf_path: str, pdf_name: str) -> Dict:
        """
        提取图片并进行分类

        Args:
            pdf_path: PDF文件路径
            pdf_name: PDF文件名

        Returns:
            分类后的图片字典
        """
        print(f"\n高级图片提取: {pdf_name}")

        # 创建输出目录
        image_dir = self.output_dir / "result" / pdf_name / "images"
        classified_dirs = {
            'figures': image_dir / "figures",
            'tables': image_dir / "tables",
            'equations': image_dir / "equations",
            'charts': image_dir / "charts",
            'other': image_dir / "other"
        }

        for dir_path in classified_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        doc = pymupdf.open(pdf_path)
        images = {
            'figures': [],
            'tables': [],
            'equations': [],
            'charts': [],
            'other': []
        }

        for page_num, page in enumerate(doc):
            # 获取图片列表
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)

                if base_image:
                    image_data = base_image["image"]
                    image_ext = base_image["ext"]

                    # 转换为RGB
                    image = Image.open(io.BytesIO(image_data))
                    if image.mode in ('RGBA', 'LA', 'P'):
                        background = Image.new('RGB', image.size, (255, 255, 255))
                        if image.mode == 'P':
                            image = image.convert('RGBA')
                        background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                        image = background

                    # 分类图片
                    category, confidence = self._classify_image(image)

                    # 保存图片
                    category_dir = classified_dirs[category]
                    image_filename = f"page{page_num + 1}_{category}_{img_index + 1}.jpg"
                    image_path = category_dir / image_filename
                    image.save(image_path, "JPEG", quality=95)

                    # 记录信息
                    image_info = {
                        'page': page_num + 1,
                        'index': img_index + 1,
                        'filename': image_filename,
                        'path': str(image_path),
                        'category': category,
                        'confidence': confidence,
                        'size': image.size
                    }

                    images[category].append(image_info)
                    print(f"  ✓ 提取{category}: {image_filename} (置信度: {confidence:.2f})")

        doc.close()

        # 统计
        total = sum(len(imgs) for imgs in images.values())
        print(f"\n总计提取 {total} 张图片:")
        for category, imgs in images.items():
            if imgs:
                print(f"  - {category}: {len(imgs)} 张")

        return images

    def _classify_image(self, image: Image.Image) -> Tuple[str, float]:
        """
        分类图片类型

        Args:
            image: PIL图像对象

        Returns:
            (类别, 置信度)
        """
        # 转换为numpy数组
        img_array = np.array(image)

        # 基本特征分析
        height, width = img_array.shape[:2]
        aspect_ratio = width / height if height > 0 else 1.0

        # 检查是否为表格（规则的网格结构）
        is_table = self._detect_table(img_array)

        # 检查是否为图表（有坐标轴、图例等）
        is_chart = self._detect_chart(img_array)

        # 检查是否为公式（包含特殊符号和结构）
        is_equation = self._detect_equation(img_array)

        # 分类决策
        if is_table:
            return 'tables', 0.85
        elif is_chart:
            return 'charts', 0.80
        elif is_equation:
            return 'equations', 0.75
        elif aspect_ratio > 0.5 and aspect_ratio < 2.0:
            # 正常比例的图片
            return 'figures', 0.70
        else:
            return 'other', 0.60

    def _detect_table(self, img_array: np.ndarray) -> bool:
        """
        检测图片是否为表格

        Args:
            img_array: 图像数组

        Returns:
            是否为表格
        """
        # 转换为灰度图
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # 检测水平线和垂直线
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)

        # 计算线条数量
        h_edges = cv2.Canny(horizontal_lines, 50, 150)
        v_edges = cv2.Canny(vertical_lines, 50, 150)

        h_count = np.sum(h_edges > 0)
        v_count = np.sum(v_edges > 0)

        # 如果有足够的水平和垂直线条，可能是表格
        return h_count > 1000 and v_count > 1000

    def _detect_chart(self, img_array: np.ndarray) -> bool:
        """
        检测图片是否为图表

        Args:
            img_array: 图像数组

        Returns:
            是否为图表
        """
        # 检测边缘
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        edges = cv2.Canny(gray, 50, 150)

        # 检测直线（Hough变换）
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                minLineLength=50, maxLineGap=10)

        if lines is not None:
            # 如果检测到多条直线，可能是图表
            return len(lines) > 5

        return False

    def _detect_equation(self, img_array: np.ndarray) -> bool:
        """
        检测图片是否为数学公式

        Args:
            img_array: 图像数组

        Returns:
            是否为公式
        """
        # 转换为灰度图
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 检测文本行
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 分析轮廓大小和分布
        if len(contours) > 0:
            areas = [cv2.contourArea(c) for c in contours]
            avg_area = np.mean(areas)

            # 公式通常有较小的平均轮廓面积（包含多个小符号）
            if avg_area < 500 and len(contours) > 10:
                return True

        return False

    def extract_figures_with_captions(self, pdf_path: str, pdf_name: str) -> List[Dict]:
        """
        提取图片及其标题

        Args:
            pdf_path: PDF文件路径
            pdf_name: PDF文件名

        Returns:
            图片和标题列表
        """
        print(f"\n提取图片和标题: {pdf_name}")

        doc = pymupdf.open(pdf_path)
        figures = []

        for page_num, page in enumerate(doc):
            # 获取文本块
            blocks = page.get_text("dict")["blocks"]

            # 查找图片和对应的标题
            for block in blocks:
                if block["type"] == 1:  # 图片块
                    bbox = block["bbox"]

                    # 在图片下方查找标题文本
                    caption = self._find_caption_near_block(blocks, bbox, block["type"])

                    if caption:
                        # 提取图片
                        image_list = page.get_images(full=True)
                        for img_index, img in enumerate(image_list):
                            # 提取并保存图片
                            # ... (图片提取代码)
                            pass

                        figures.append({
                            'page': page_num + 1,
                            'bbox': bbox,
                            'caption': caption,
                            'type': self._guess_figure_type(caption)
                        })

        doc.close()
        return figures

    def _find_caption_near_block(self, blocks: List, bbox: Tuple, block_type: int) -> str:
        """
        在块附近查找标题文本

        Args:
            blocks: 所有文本块
            bbox: 块的边界框
            block_type: 块类型

        Returns:
            标题文本
        """
        x0, y0, x1, y1 = bbox

        # 在图片下方查找文本
        for block in blocks:
            if block["type"] == 0:  # 文本块
                block_bbox = block["bbox"]
                bx0, by0, bx1, by1 = block_bbox

                # 检查是否在图片下方且接近
                if (by0 > y1 and by0 - y1 < 100 and  # 在下方100像素内
                    bx0 < x1 and bx1 > x0):  # 水平重叠
                    # 提取文本
                    for line in block.get("lines", []):
                        line_text = "".join(span["text"] for span in line.get("spans", []))

                        # 检查是否为标题（以Figure、Table等开头）
                        if line_text.startswith(('Figure', 'Fig', 'Table', 'Chart')):
                            return line_text

        return ""

    def _guess_figure_type(self, caption: str) -> str:
        """
        根据标题猜测图片类型

        Args:
            caption: 标题文本

        Returns:
            图片类型
        """
        caption_lower = caption.lower()

        if 'table' in caption_lower:
            return 'table'
        elif 'figure' in caption_lower or 'fig' in caption_lower:
            if 'graph' in caption_lower or 'plot' in caption_lower:
                return 'chart'
            else:
                return 'figure'
        elif 'chart' in caption_lower or 'graph' in caption_lower:
            return 'chart'
        elif 'equation' in caption_lower or 'eq' in caption_lower:
            return 'equation'
        else:
            return 'unknown'


class VLAImageExtractor(AdvancedImageExtractor):
    """使用VLA模型的图片提取器"""

    def __init__(self, output_dir: str = "output", model_name: str = "pix2struct"):
        """
        初始化VLA提取器

        Args:
            output_dir: 输出文件夹
            model_name: 模型名称
        """
        super().__init__(output_dir)
        self.model_name = model_name

    def extract_with_vla(self, pdf_path: str, pdf_name: str) -> List[Dict]:
        """
        使用VLA模型提取和理解图片

        Args:
            pdf_path: PDF文件路径
            pdf_name: PDF文件名

        Returns:
            图片理解结果
        """
        print(f"\n使用VLA模型提取: {pdf_name}")

        try:
            if self.model_name == "pix2struct":
                return self._extract_with_pix2struct(pdf_path, pdf_name)
            elif self.model_name == "nougat":
                return self._extract_with_nougat(pdf_path, pdf_name)
            else:
                print(f"  ⚠ 未知的VLA模型: {self.model_name}")
                return []

        except Exception as e:
            print(f"  ⚠ VLA提取失败: {e}")
            return []

    def _extract_with_pix2struct(self, pdf_path: str, pdf_name: str) -> List[Dict]:
        """
        使用Pix2Struct提取图表

        Args:
            pdf_path: PDF文件路径
            pdf_name: PDF文件名

        Returns:
            提取结果
        """
        try:
            from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

            print("  → 加载Pix2Struct模型...")
            model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-screen2words-base")
            processor = Pix2StructProcessor.from_pretrained("google/pix2struct-screen2words-base")

            # 提取PDF页面为图片
            doc = pymupdf.open(pdf_path)
            results = []

            for page_num, page in enumerate(doc):
                # 渲染页面
                pix = page.get_pixmap(dpi=200)
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))

                # 处理图像
                inputs = processor(images=image, text="<screenshot>", return_tensors="pt")
                predictions = model.generate(**inputs)
                caption = processor.decode(predictions[0], skip_special_tokens=True)

                results.append({
                    'page': page_num + 1,
                    'caption': caption,
                    'type': 'vla_understanding'
                })

                print(f"  ✓ 页面 {page_num + 1}: {caption[:50]}...")

            doc.close()
            return results

        except ImportError:
            print("  ⚠ transformers库未安装")
            return []
        except Exception as e:
            print(f"  ⚠ Pix2Struct处理失败: {e}")
            return []

    def _extract_with_nougat(self, pdf_path: str, pdf_name: str) -> List[Dict]:
        """
        使用Nougat OCR提取学术内容

        Args:
            pdf_path: PDF文件路径
            pdf_name: PDF文件名

        Returns:
            提取结果
        """
        try:
            import subprocess

            print("  → 使用Nougat OCR...")

            # 调用nougat命令
            result = subprocess.run(
                ['nougat', pdf_path, '-o', str(self.output_dir / "result" / pdf_name)],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                print("  ✓ Nougat处理成功")
                # 解析输出
                return []
            else:
                print(f"  ⚠ Nougat处理失败: {result.stderr}")
                return []

        except FileNotFoundError:
            print("  ⚠ Nougat未安装")
            return []
        except Exception as e:
            print(f"  ⚠ Nougat处理失败: {e}")
            return []
