"""
数学表达式提取器
识别和转换PDF中的数学表达式为LaTeX格式
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
import pymupdf


class MathExtractor:
    """数学表达式提取器"""

    def __init__(self):
        """初始化提取器"""
        # 数学表达式模式
        self.inline_patterns = [
            r'\$[^$]+\$',  # $...$
            r'\\[a-zA-Z]+\{[^}]*\}',  # LaTeX命令
            r'[a-z]\_[0-9]+',  # 下标 x_1
            r'[a-z]\^\{?[0-9]+\}?',  # 上标 x^2
            r'\\[a-zA-Z]+',  # LaTeX命令
            r'∫|∑|∏|√|α|β|γ|θ|λ|μ|σ|ω',  # 数学符号
        ]

        self.display_patterns = [
            r'\$\$[^$]+\$\$',  # $$...$$
            r'\\\[.*?\\\]',  # \[...\]
            r'\\begin\{equation\}.*?\\end\{equation\}',
            r'\\begin\{align\}.*?\\end\{align\}',
        ]

    def extract_math(self, text: str) -> Tuple[str, List[Dict]]:
        """
        从文本中提取数学表达式

        Args:
            text: 输入文本

        Returns:
            (清理后的文本, 数学表达式列表)
        """
        math_expressions = []

        # 提取行内公式
        for pattern in self.inline_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                expr = match.group(0)
                math_expressions.append({
                    'type': 'inline',
                    'original': expr,
                    'latex': self._to_latex(expr),
                    'position': match.span()
                })

        # 提取显示公式
        for pattern in self.display_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                expr = match.group(0)
                math_expressions.append({
                    'type': 'display',
                    'original': expr,
                    'latex': self._to_latex(expr),
                    'position': match.span()
                })

        # 清理文本，移除原始数学表达式
        cleaned_text = text
        for expr_info in reversed(math_expressions):
            start, end = expr_info['position']
            placeholder = f"MATHPLACEHOLDER{expr_info['type'].upper()}"
            cleaned_text = cleaned_text[:start] + placeholder + cleaned_text[end:]

        return cleaned_text, math_expressions

    def extract_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        从PDF中提取数学表达式

        Args:
            pdf_path: PDF文件路径

        Returns:
            数学表达式列表（按页面组织）
        """
        doc = pymupdf.open(pdf_path)
        all_math = []

        for page_num, page in enumerate(doc):
            # 获取文本块
            blocks = page.get_text("dict")["blocks"]

            page_math = {
                'page': page_num + 1,
                'expressions': []
            }

            for block in blocks:
                if block["type"] == 0:  # 文本块
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span["text"]
                            # 检测数学表达式
                            if self._is_math_expression(text):
                                page_math['expressions'].append({
                                    'text': text,
                                    'bbox': span["bbox"],
                                    'font': span.get("font", ""),
                                    'latex': self._text_to_latex(text)
                                })

            all_math.append(page_math)

        doc.close()
        return all_math

    def _is_math_expression(self, text: str) -> bool:
        """
        判断文本是否为数学表达式

        Args:
            text: 输入文本

        Returns:
            是否为数学表达式
        """
        # 数学符号
        math_symbols = {'∑', '∏', '∫', '√', '∞', '±', '∓', '×', '÷', '∂', '∇', 'Δ', 'α', 'β', 'γ', 'θ', 'λ', 'μ', 'σ', 'ω', 'π'}

        # 检查是否包含数学符号
        if any(char in text for char in math_symbols):
            return True

        # 检查数学模式
        math_patterns = [
            r'[a-z]\_[0-9]+',  # 下标
            r'[a-z]\^[\{]?[0-9]+[\}]?',  # 上标
            r'\\[a-zA-Z]+\{',  # LaTeX命令
            r'\$.*?\$',  # LaTeX公式
            r'[0-9]+\s*[+\-*/]\s*[0-9]+',  # 算术表达式
            r'[a-z]\s*=\s*[a-z0-9+\-*/]+',  # 等式
        ]

        for pattern in math_patterns:
            if re.search(pattern, text):
                return True

        return False

    def _to_latex(self, expr: str) -> str:
        """
        将数学表达式转换为LaTeX格式

        Args:
            expr: 数学表达式

        Returns:
            LaTeX字符串
        """
        # 如果已经是LaTeX格式，直接返回
        if expr.startswith('$') or expr.startswith('\\'):
            return expr

        # 转换为LaTeX
        latex = expr

        # 转换常见数学符号
        symbol_map = {
            'α': r'\\alpha',
            'β': r'\\beta',
            'γ': r'\\gamma',
            'θ': r'\\theta',
            'λ': r'\\lambda',
            'μ': r'\\mu',
            'σ': r'\\sigma',
            'ω': r'\\omega',
            'π': r'\\pi',
            '∫': r'\\int',
            '∑': r'\\sum',
            '∏': r'\\prod',
            '√': r'\\sqrt',
            '∞': r'\\infty',
            '±': r'\\pm',
            '∓': r'\\mp',
            '×': r'\\times',
            '÷': r'\\div',
            '∂': r'\\partial',
            '∇': r'\\nabla',
            'Δ': r'\\Delta',
        }

        for symbol, latex_cmd in symbol_map.items():
            latex = latex.replace(symbol, latex_cmd)

        # 转换上下标
        # 下标: x_1 -> x_{1}
        latex = re.sub(r'([a-zA-Z])_([0-9]+)', r'\1_{\2}', latex)
        # 上标: x^2 -> x^{2}
        latex = re.sub(r'([a-zA-Z])\^([0-9]+)', r'\1^{\2}', latex)

        # 转换分数: a/b -> \frac{a}{b}
        latex = re.sub(r'([a-zA-Z0-9]+)/([a-zA-Z0-9]+)', r'\\frac{\1}{\2}', latex)

        # 添加公式定界符
        if not latex.startswith('$'):
            latex = f'${latex}$'

        return latex

    def _text_to_latex(self, text: str) -> str:
        """
        将文本中的数学表达式转换为LaTeX

        Args:
            text: 包含数学表达式的文本

        Returns:
            LaTeX字符串
        """
        return self._to_latex(text)

    def enhance_text_with_math(self, text: str) -> str:
        """
        增强文本，将数学表达式转换为LaTeX格式

        Args:
            text: 原始文本

        Returns:
            增强后的文本（包含LaTeX公式）
        """
        lines = text.split('\n')
        enhanced_lines = []

        for line in lines:
            if self._is_math_expression(line):
                # 转换为LaTeX
                latex = self._to_latex(line)
                enhanced_lines.append(latex)
            else:
                enhanced_lines.append(line)

        return '\n'.join(enhanced_lines)


class AdvancedMathExtractor(MathExtractor):
    """高级数学表达式提取器（支持OCR和图像识别）"""

    def extract_from_image(self, image_path: str) -> List[Dict]:
        """
        从图像中提取数学表达式

        Args:
            image_path: 图像文件路径

        Returns:
            数学表达式列表
        """
        try:
            from ..latex_ocr_bridge import ensure_latex_ocr_on_path

            ensure_latex_ocr_on_path()
            from pix2tex.cli import LatexOCR

            model = LatexOCR()
            from PIL import Image

            img = Image.open(image_path)
            latex_code = model(img)

            return [{
                'type': 'image',
                'latex': latex_code,
                'source': image_path
            }]

        except ImportError:
            print("  ⚠ pix2tex未安装，使用基本识别")
            return []
        except Exception as e:
            print(f"  ⚠ 图像识别失败: {e}")
            return []

    def extract_with_mathpix(self, pdf_path: str, api_key: str = None) -> List[Dict]:
        """
        使用Mathpix API提取数学表达式

        Args:
            pdf_path: PDF文件路径
            api_key: Mathpix API密钥

        Returns:
            数学表达式列表
        """
        if not api_key:
            print("  ⚠ 需要Mathpix API密钥")
            return []

        try:
            import requests

            # 这里需要实现Mathpix API调用
            # 具体实现需要参考Mathpix API文档
            print("  ⚠ Mathpix集成需要API密钥和具体实现")
            return []

        except ImportError:
            print("  ⚠ requests库未安装")
            return []
