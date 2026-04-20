"""
PDF图片提取器
提取PDF中的所有图片并转换为JPG格式
"""

from pathlib import Path
from typing import Dict, List, Optional, Set

import pymupdf
from PIL import Image
import io


class ImageExtractor:
    """PDF图片提取器"""

    def __init__(self, output_dir: str = "output"):
        """
        初始化提取器

        Args:
            output_dir: 输出文件夹
        """
        self.output_dir = Path(output_dir)

    def extract(
        self,
        pdf_path: str,
        pdf_name: str,
        pages: Optional[List[int]] = None,
    ) -> List[Dict]:
        """
        提取PDF中的所有图片

        Args:
            pdf_path: PDF文件路径
            pdf_name: PDF文件名（不含扩展名）
            pages: 若指定，仅提取这些页（1-based）；None 表示全部页

        Returns:
            图片信息列表
        """
        print(f"\n提取图片: {pdf_name}")

        # 创建输出目录
        image_dir = self.output_dir / "result" / pdf_name / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        doc = pymupdf.open(pdf_path)
        images = []
        image_counter = 0
        page_filter: Optional[Set[int]] = set(pages) if pages else None

        for page_num, page in enumerate(doc):
            if page_filter is not None and (page_num + 1) not in page_filter:
                continue
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]

                # 提取图片
                base_image = doc.extract_image(xref)

                if base_image:
                    image_counter += 1
                    image_data = base_image["image"]
                    image_ext = base_image["ext"]

                    # 转换为JPG
                    image = Image.open(io.BytesIO(image_data))

                    # 处理透明度（RGBA转RGB）
                    if image.mode in ('RGBA', 'LA', 'P'):
                        background = Image.new('RGB', image.size, (255, 255, 255))
                        if image.mode == 'P':
                            image = image.convert('RGBA')
                        background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                        image = background

                    # 保存为JPG
                    image_filename = f"page{page_num + 1}_img{img_index + 1}.jpg"
                    image_path = image_dir / image_filename
                    image.save(image_path, "JPEG", quality=95)

                    image_info = {
                        'page': page_num + 1,
                        'index': img_index + 1,
                        'filename': image_filename,
                        'path': str(image_path),
                        'size': image.size,
                        'xref': xref
                    }
                    images.append(image_info)

                    print(f"  ✓ 提取图片: {image_filename}")

        doc.close()

        print(f"\n总计提取 {len(images)} 张图片")
        return images

    def extract_page_as_image(self, pdf_path: str, page_num: int, output_path: str):
        """
        将PDF页面转换为图片（用于质量对比）

        Args:
            pdf_path: PDF文件路径
            page_num: 页码（从1开始）
            output_path: 输出图片路径
        """
        doc = pymupdf.open(pdf_path)

        if 0 < page_num <= len(doc):
            page = doc[page_num - 1]
            pix = page.get_pixmap(dpi=150)
            pix.save(output_path)
            doc.close()
            return True

        doc.close()
        return False
