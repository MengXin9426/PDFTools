"""
PDF文件搜索器
扫描指定文件夹，识别所有英文PDF文件
"""

import os
from pathlib import Path
from typing import List, Dict
import pymupdf


class PDFSearcher:
    """PDF文件搜索器"""

    def __init__(self, input_dir: str = "input"):
        """
        初始化搜索器

        Args:
            input_dir: 输入文件夹路径
        """
        self.input_dir = Path(input_dir)
        self.pdf_files = []

    def search_pdfs(self) -> List[Dict]:
        """
        搜索所有PDF文件并检测语言

        Returns:
            PDF文件信息列表
        """
        if not self.input_dir.exists():
            self.input_dir.mkdir(parents=True)
            print(f"创建输入文件夹: {self.input_dir}")
            return []

        pdf_files = list(self.input_dir.glob("*.pdf"))
        results = []

        for pdf_path in pdf_files:
            info = self._analyze_pdf(pdf_path)
            if info and info['is_english']:
                results.append(info)
                print(f"✓ 找到英文PDF: {pdf_path.name}")
            elif info:
                print(f"⊗ 跳过非英文PDF: {pdf_path.name}")

        self.pdf_files = results
        return results

    def _analyze_pdf(self, pdf_path: Path) -> Dict:
        """
        分析PDF文件信息

        Args:
            pdf_path: PDF文件路径

        Returns:
            文件信息字典
        """
        try:
            doc = pymupdf.open(str(pdf_path))

            # 提取前几页文本用于语言检测
            sample_text = ""
            max_pages = min(3, len(doc))
            for i in range(max_pages):
                sample_text += doc[i].get_text()

            # 简单的英文检测：检查英文字符比例
            is_english = self._detect_english(sample_text)

            return {
                'path': str(pdf_path),
                'name': pdf_path.stem,
                'pages': len(doc),
                'is_english': is_english,
                'size': pdf_path.stat().st_size
            }

        except Exception as e:
            print(f"分析文件失败 {pdf_path.name}: {e}")
            return None

    def _detect_english(self, text: str) -> bool:
        """
        检测文本是否为英文

        Args:
            text: 待检测文本

        Returns:
            是否为英文
        """
        if not text or len(text) < 50:
            return True  # 默认为英文

        # 计算ASCII字符比例
        ascii_chars = sum(1 for c in text if ord(c) < 128 and c.isalpha())
        total_chars = sum(1 for c in text if c.isalpha())

        if total_chars == 0:
            return True

        ascii_ratio = ascii_chars / total_chars

        # 如果ASCII字符比例超过85%，认为是英文
        return ascii_ratio > 0.85
