"""
使用阿里通义千问（Qwen）翻译学术段落，保留公式行不译。

性能优化：将多个 chunk 合并为大批次一次性翻译，大幅减少 API 调用次数。
"""

from typing import Dict, List, Optional

from .glossary import get_abbreviation_mapping
from ..qwen_enhancer.enhancer import QwenEnhancer

_FORMULA_PLACEHOLDER_PREFIX = "<<FORMULA_LINE_"
_FORMULA_PLACEHOLDER_SUFFIX = ">>"
_CHUNK_SEPARATOR = "\n===CHUNK_BREAK===\n"
_BATCH_CHAR_LIMIT = 6000
_BATCH_CHUNK_LIMIT = 30


class QwenTranslator:
    """LLM 翻译器（超级批量模式，支持 Qwen 云端 / vLLM 本地）。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        backend: Optional[str] = None,
        source_lang: str = "en",
        target_lang: str = "zh-CN",
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.abbreviation_map = get_abbreviation_mapping()
        self.qwen = QwenEnhancer(api_key, model=model, backend=backend)

    @staticmethod
    def _is_formula_only_line(line: str) -> bool:
        s = (line or "").strip()
        if len(s) >= 4 and s.startswith("$$") and s.endswith("$$"):
            return True
        if len(s) >= 2 and s.startswith("$") and s.endswith("$") and s.count("$") == 2:
            return True
        return False

    def _protect_formulas(self, lines: List[str]) -> tuple:
        """将公式行替换为占位符，返回 (处理后的行列表, 公式映射)。"""
        protected: List[str] = []
        formula_map: Dict[int, str] = {}
        for idx, line in enumerate(lines):
            if not line or not line.strip():
                protected.append("")
            elif self._is_formula_only_line(line):
                ph = f"{_FORMULA_PLACEHOLDER_PREFIX}{idx}{_FORMULA_PLACEHOLDER_SUFFIX}"
                protected.append(ph)
                formula_map[idx] = line
            else:
                protected.append(line)
        return protected, formula_map

    def _restore_formulas(self, translated: List[str], formula_map: Dict[int, str], orig_len: int) -> List[str]:
        """将翻译结果中的公式占位符还原。"""
        if len(translated) < orig_len:
            translated.extend([""] * (orig_len - len(translated)))
        elif len(translated) > orig_len:
            translated = translated[:orig_len]
        for idx, orig in formula_map.items():
            if idx < len(translated):
                translated[idx] = orig
        return translated

    def _extract_translatable(self, chunk: Dict) -> Optional[tuple]:
        """从 chunk 中提取需要翻译的文本。返回 (text, chunk_type) 或 None。"""
        ctype = chunk.get("type", "")
        if ctype in ("image", "latex_raw"):
            return None
        if ctype == "header":
            text = (chunk.get("content") or [""])[0] or ""
            if text.strip():
                return (text, "header")
            return None
        if ctype == "text_group":
            lines: List[str] = []
            for tc in chunk.get("chunks", []):
                for line in tc.get("content", []):
                    lines.append(line)
            if lines:
                return ("\n".join(lines), "text_group")
        return None

    def _translate_single_text(self, text: str, context: Optional[str] = None) -> str:
        result = self.qwen.enhance_text_translation(text, context)
        if result is None or not str(result).strip():
            return text
        return str(result)

    def _translate_mega_batch(self, batch_texts: List[str]) -> List[str]:
        """将多个文本段用分隔符合并后一次性翻译，再按分隔符拆回。"""
        if not batch_texts:
            return []
        if len(batch_texts) == 1:
            return [self._translate_single_text(batch_texts[0])]

        combined = _CHUNK_SEPARATOR.join(batch_texts)
        prompt = (
            "请将以下学术文本翻译成中文，要求：\n"
            "1. 保持专业术语的准确性，符合中文学术表达习惯\n"
            "2. 专有名词、缩写（如 DSSS、OFDM、MIMO、IEEE 等）保持原样不翻译\n"
            "3. 数学公式、LaTeX 命令、占位符行保持原样不改\n"
            "4. 变量名、函数名、单位符号保持原样\n"
            "5. **严格保留文本中的 ===CHUNK_BREAK=== 分隔符，位置和数量不变**\n\n"
            f"待翻译文本：\n{combined}\n\n"
            "请只返回翻译结果，保持分隔符。"
        )
        try:
            resp = self.qwen.client.chat.completions.create(
                model=self.qwen.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一个专业的学术翻译助手，擅长翻译英文科技论文。"
                            "翻译时必须严格保留所有 ===CHUNK_BREAK=== 分隔符。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=8192,
                **self.qwen._extra_create_kwargs(),
            )
            raw = resp.choices[0].message.content or ""
            raw = raw.strip()
            parts = raw.split("===CHUNK_BREAK===")
            parts = [p.strip() for p in parts]

            if len(parts) == len(batch_texts):
                return parts

            if len(parts) < len(batch_texts):
                parts.extend(batch_texts[len(parts):])
            else:
                parts = parts[:len(batch_texts)]
            return parts

        except Exception as e:
            print(f"  ⚠ 超级批量翻译失败 ({len(batch_texts)} 段)，回退逐段: {e}")
            return [self._translate_single_text(t) for t in batch_texts]

    def translate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        print(f"\n使用 {self.qwen.backend.upper()} 后端翻译...")
        total = len(chunks)
        # 计算总待翻译字符数
        total_chars = 0
        for chunk in chunks:
            ext = self._extract_translatable(chunk)
            if ext:
                text, _ = ext
                total_chars += len(text)
        print(f"总段数: {total}, 总待翻译字符数: {total_chars}")

        processed_chars = 0
        pending_indices: List[int] = []
        pending_texts: List[str] = []
        pending_types: List[str] = []
        pending_char_count = 0

        translated_chunks: List[Dict] = [None] * total  # type: ignore

        def _flush_batch():
            nonlocal pending_indices, pending_texts, pending_types, pending_char_count, processed_chars
            if not pending_texts:
                return
            n = len(pending_texts)
            print(f"  → 批量翻译 {n} 段 ({pending_char_count} 字符)...")
            results = self._translate_mega_batch(pending_texts)

            for idx, text, ctype, result in zip(pending_indices, pending_texts, pending_types, results):
                chunk = chunks[idx]
                if ctype == "header":
                    translated_chunks[idx] = {
                        "page": chunk.get("page", 0),
                        "type": "header",
                        "content": [result],
                        "original": text,
                    }
                elif ctype == "text_group":
                    orig_lines = text.split("\n")
                    trans_lines = result.split("\n")

                    protected, formula_map = self._protect_formulas(orig_lines)
                    trans_lines = self._restore_formulas(trans_lines, formula_map, len(orig_lines))

                    line_idx = 0
                    translated_group: List[Dict] = []
                    for tc in chunk.get("chunks", []):
                        content = tc.get("content", [])
                        n_lines = len(content)
                        chunk_trans = trans_lines[line_idx:line_idx + n_lines]
                        if len(chunk_trans) < n_lines:
                            chunk_trans.extend([""] * (n_lines - len(chunk_trans)))
                        translated_group.append({
                            "page": tc.get("page", 0),
                            "content": chunk_trans,
                            "original": content,
                        })
                        line_idx += n_lines
                    translated_chunks[idx] = {"type": "text_group", "chunks": translated_group}

                    print(f"  ✓ 批量完成 {n} 段")
                    # 累加已处理字符数并显示进度条
                    nonlocal processed_chars
                    batch_char_sum = sum(len(t) for t in pending_texts)
                    processed_chars += batch_char_sum
                    percent = processed_chars / total_chars if total_chars > 0 else 0
                    bar_len = 40
                    filled = int(bar_len * percent)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(f"  进度: |{bar}| {percent*100:.1f}% ({processed_chars}/{total_chars} 字符)")

                    pending_indices.clear()
                    pending_texts.clear()
                    pending_types.clear()
                    pending_char_count = 0

        for i, chunk in enumerate(chunks):
            ctype = chunk.get("type", "")

            if ctype in ("image", "latex_raw"):
                translated_chunks[i] = chunk
                continue

            ext = self._extract_translatable(chunk)
            if ext is None:
                translated_chunks[i] = chunk
                continue

            text, t_type = ext
            text_len = len(text)

            should_flush = (
                pending_texts
                and (
                    pending_char_count + text_len > _BATCH_CHAR_LIMIT
                    or len(pending_texts) >= _BATCH_CHUNK_LIMIT
                )
            )
            if should_flush:
                _flush_batch()

            pending_indices.append(i)
            pending_texts.append(text)
            pending_types.append(t_type)
            pending_char_count += text_len

        _flush_batch()
        # 最终100%进度条（确保显示）
        if total_chars > 0:
            bar_len = 40
            filled = bar_len
            bar = "█" * filled
            print(f"  进度: |{bar}| 100.0% ({total_chars}/{total_chars} 字符)")
        
        result_chunks = [c for c in translated_chunks if c is not None]
        skipped = total - len(result_chunks)
        if skipped:
            print(f"  ℹ 跳过 {skipped} 个空/无效 chunk")
        return result_chunks
