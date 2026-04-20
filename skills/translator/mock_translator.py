"""
模拟翻译器 - 用于测试
不需要网络连接
"""

from .translator import Translator
from .glossary import protect_abbreviations, restore_abbreviations


class MockTranslator(Translator):
    """模拟翻译器 - 用于测试"""

    def _translate_text(self, text: str) -> str:
        """
        模拟翻译（只添加前缀，不实际翻译）

        Args:
            text: 待翻译文本

        Returns:
            模拟翻译后的文本
        """
        if not text or not text.strip():
            return text

        try:
            # 保护缩写词
            text_with_abbr_protection, abbr_placeholders = protect_abbreviations(text)

            # 保护LaTeX命令
            protected_text = self._protect_latex(text_with_abbr_protection)

            # 模拟翻译（添加中文前缀）
            mock_translated = f"[中文]{protected_text}"

            # 恢复LaTeX命令
            mock_translated = self._restore_latex(mock_translated)

            # 恢复缩写词
            mock_translated = restore_abbreviations(mock_translated, abbr_placeholders)

            return mock_translated

        except Exception as e:
            print(f"  ⚠ 模拟翻译失败: {e}")
            return f"[中文]{text}"
