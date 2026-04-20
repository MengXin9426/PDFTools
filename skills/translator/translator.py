"""
英中翻译器
翻译文本块并保留LaTeX命令和专有名词
"""

import re
from typing import List, Dict

from .glossary import protect_abbreviations, restore_abbreviations, get_abbreviation_mapping

# 与 Kimi 等引擎共用的术语表（浅拷贝使用）
TRANSLATION_GLOSSARY = {
    "machine learning": "机器学习",
    "deep learning": "深度学习",
    "neural network": "神经网络",
    "algorithm": "算法",
    "optimization": "优化",
    "convergence": "收敛",
    "gradient": "梯度",
    "loss function": "损失函数",
    "backpropagation": "反向传播",
    "waveform design": "波形设计",
    "mutual information": "互信息",
    "detection": "检测",
    "estimation": "估计",
    "clutter": "杂波",
    "Gaussian mixture": "高斯混合",
}


class Translator:
    """英中翻译器"""

    def __init__(self, source_lang: str = 'en', target_lang: str = 'zh-CN'):
        """
        初始化翻译器

        Args:
            source_lang: 源语言
            target_lang: 目标语言
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        try:
            from deep_translator import GoogleTranslator
        except ImportError as e:
            raise ImportError(
                "使用 Translator（Google 翻译）需要安装依赖: pip install deep-translator"
            ) from e
        self.translator = GoogleTranslator(source=source_lang, target=target_lang)

        self.glossary = dict(TRANSLATION_GLOSSARY)

        # 加载缩写词映射
        self.abbreviation_map = get_abbreviation_mapping()

    def translate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        翻译文本块

        Args:
            chunks: 分块列表

        Returns:
            翻译后的分块列表
        """
        print("\n翻译文本...")

        translated_chunks = []

        for i, chunk in enumerate(chunks):
            if chunk['type'] in ('image', 'latex_raw'):
                translated_chunks.append(chunk)

            elif chunk['type'] == 'header':
                translated = self._translate_text(chunk['content'][0])
                translated_chunks.append({
                    'page': chunk['page'],
                    'type': 'header',
                    'content': [translated],
                    'original': chunk['content'][0]
                })
                print(f"  ✓ 翻译标题 {i + 1}/{len(chunks)}")

            elif chunk['type'] == 'text_group':
                translated_group = []

                for text_chunk in chunk['chunks']:
                    translated_lines = []
                    for line in text_chunk.get('content', []):
                        if self._is_formula_only_line(line):
                            translated_lines.append(line)
                        else:
                            translated = self._translate_text(line)
                            translated_lines.append(translated)

                    translated_group.append({
                        'page': text_chunk['page'],
                        'content': translated_lines,
                        'original': text_chunk.get('content', [])
                    })

                translated_chunks.append({
                    'type': 'text_group',
                    'chunks': translated_group
                })
                print(f"  ✓ 翻译文本块 {i + 1}/{len(chunks)}")

        return translated_chunks

    @staticmethod
    def _is_formula_only_line(line: str) -> bool:
        """整行仅为 ``$$...$$`` 或 ``$...$`` 的纯公式行不送翻译引擎。"""
        s = (line or "").strip()
        if len(s) >= 4 and s.startswith("$$") and s.endswith("$$"):
            return True
        if len(s) >= 2 and s.startswith("$") and s.endswith("$") and s.count("$") == 2:
            return True
        return False

    def _translate_text(self, text: str) -> str:
        """
        翻译单行文本

        Args:
            text: 待翻译文本

        Returns:
            翻译后的文本
        """
        if not text or not text.strip():
            return text

        if self._is_formula_only_line(text):
            return text

        # 1. 保护缩写词（专有名词）
        text_with_abbr_protection, abbr_placeholders = protect_abbreviations(text)

        # 2. 保护LaTeX命令
        protected_text = self._protect_latex(text_with_abbr_protection)

        try:
            # 3. 翻译
            translated = self.translator.translate(protected_text)

            # 4. 恢复LaTeX命令
            translated = self._restore_latex(translated)

            # 5. 恢复缩写词
            translated = restore_abbreviations(translated, abbr_placeholders)

            # 6. 应用术语词典
            translated = self._apply_glossary(translated)

            return translated

        except Exception as e:
            print(f"  ⚠ 翻译失败: {e}")
            return text

    def _protect_latex(self, text: str) -> str:
        """
        保护LaTeX命令不被翻译

        Args:
            text: 原始文本

        Returns:
            保护后的文本
        """
        # 匹配LaTeX命令
        latex_pattern = r'(\\[a-zA-Z]+|\\[\[\]\{\}]|\$.*?\$|\\begin\{.*?\}|\\end\{.*?\})'

        # 创建占位符映射
        self._latex_placeholders = {}
        placeholder_count = 0

        def replace_latex(match):
            nonlocal placeholder_count
            placeholder = f"LATEXPLACEHOLDER{placeholder_count}LATEXPLACEHOLDER"
            self._latex_placeholders[placeholder] = match.group(1)
            placeholder_count += 1
            return placeholder

        protected = re.sub(latex_pattern, replace_latex, text)
        return protected

    def _restore_latex(self, text: str) -> str:
        """
        恢复LaTeX命令

        Args:
            text: 包含占位符的文本

        Returns:
            恢复后的文本
        """
        for placeholder, latex_cmd in self._latex_placeholders.items():
            text = text.replace(placeholder, latex_cmd)

        return text

    def _apply_glossary(self, text: str) -> str:
        """
        应用术语词典

        Args:
            text: 翻译后的文本

        Returns:
            应用术语后的文本
        """
        for en_term, zh_term in self.glossary.items():
            text = text.replace(en_term, zh_term)

        return text
