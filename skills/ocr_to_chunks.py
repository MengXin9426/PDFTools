"""将 Qwen-VL-OCR 产出的 LaTeX/Markdown 解析为 pipeline 标准 chunk 格式。

chunk 类型：
    - header:     章节标题（纯文本，需翻译）
    - text_group: 正文段落（纯文本，需翻译；行内 $...$ 由翻译器保护）
    - latex_raw:  equation/align/tabular 等环境，原样写入 .tex 不翻译
    - image:      figure 环境位置，用页面渲染图替换

核心原则：OCR 产出的 LaTeX 环境必须原样保留到 .tex 文件，不做字符转义。
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

_SECTION_RE = re.compile(
    r'\\(?:section|subsection|subsubsection)\*?\{(.+?)\}',
    re.DOTALL,
)

def _extract_balanced_caption(text: str) -> str:
    """从含 \\caption{...} 的文本中提取 caption，正确处理嵌套花括号。"""
    idx = text.find(r'\caption{')
    if idx < 0:
        return ""
    start = idx + len(r'\caption{')
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    return text[start:i - 1].strip() if depth == 0 else text[start:i].strip()

_INLINE_DOUBLE_DOLLAR_RE = re.compile(r'\$\$([^$\n]+?)\$\$')


def _normalize_inline_math(text: str) -> str:
    """将句中的 $$...$$ 转为行内 $...$。

    Qwen-VL-OCR 对行内公式也使用 $$...$$，需要区分：
    - 独占一行的 $$...$$ => 保留为 display math
    - 句中的 $$short_expr$$ => 转为 $...$
    """
    def replacer(m: re.Match) -> str:
        start, end = m.start(), m.end()
        before_char = text[start - 1] if start > 0 else '\n'
        after_char = text[end] if end < len(text) else '\n'

        if before_char == '\n' and after_char == '\n':
            return m.group(0)

        return f'${m.group(1)}$'

    return _INLINE_DOUBLE_DOLLAR_RE.sub(replacer, text)


def _strip_code_fences(text: str) -> str:
    text = re.sub(r'^```\s*latex\s*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n```\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


_WATERMARK_RE = re.compile(
    r'^(万方数据|维普资讯|中国知网|CNKI)\s*$', re.MULTILINE | re.IGNORECASE,
)
_HEADER_FOOTER_RE = re.compile(
    r'^.*(?:Front(?:iers)?\s+(?:of\s+)?Inform|ISSN\s+\d{4}|'
    r'jzus@zju\.edu\.cn|E-mail:\s+jzus@|'
    r'www\.\w+\.(?:zju|cae)\.(?:cn|edu)|springerlink\.com)\b.*$',
    re.MULTILINE | re.IGNORECASE,
)
_PAGE_NUMBER_LINE_RE = re.compile(
    r'^\s*\d{1,5}\s*$', re.MULTILINE,
)
_FLUSHRIGHT_BLOCK_RE = re.compile(
    r'\\begin\{flushright\}[\s\S]*?\\end\{flushright\}',
)
_JOURNAL_BADGE_RE = re.compile(
    r'^\\textbf\{(?:FITEE|JZUS)\}\s*$', re.MULTILINE,
)
_DOI_CLC_RE = re.compile(
    r'^.*(?:https?://doi\.org/|CLC\s+number:).*$', re.MULTILINE | re.IGNORECASE,
)
_AUTHOR_HEADER_RE = re.compile(
    r'^[A-Z][a-z]+ et al\.\s*/\s*\w.*\d{4}\s+\d+\(\d+\):\d+-\d+.*$',
    re.MULTILINE,
)
_JOURNAL_TABULAR_RE = re.compile(
    r'\\begin\{tabular\}\{[^}]*\}\s*\n'
    r'(?:(?:.*(?:ISSN|Front(?:iers)?|www\.|E-mail:|doi\.org|\d{4}-\d{4}|springer).*\n?)+)'
    r'\\end\{tabular\}',
    re.IGNORECASE,
)


def _clean_page_junk(content: str) -> str:
    """移除 OCR 页面中的页眉、页脚、水印、期刊标识等非正文内容。"""
    content = _WATERMARK_RE.sub('', content)
    content = _JOURNAL_BADGE_RE.sub('', content)
    content = _AUTHOR_HEADER_RE.sub('', content)
    content = _HEADER_FOOTER_RE.sub('', content)
    content = _FLUSHRIGHT_BLOCK_RE.sub('', content)
    content = _JOURNAL_TABULAR_RE.sub('', content)
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content.strip()


def _split_pages(full_md: str) -> List[Tuple[int, str]]:
    parts = re.split(r'##\s*Page\s+(\d+)', full_md)
    pages: List[Tuple[int, str]] = []
    i = 1
    while i < len(parts) - 1:
        pages.append((int(parts[i]), parts[i + 1].strip()))
        i += 2
    return pages


def _is_header_line(line: str) -> bool:
    return bool(_SECTION_RE.match(line.strip()))


def _extract_header_text(line: str) -> str:
    m = _SECTION_RE.match(line.strip())
    return m.group(1).strip() if m else line.strip()


_JUNK_LINE_PATTERNS = [
    re.compile(r'^\\begin\{center\}$'),
    re.compile(r'^\\end\{center\}$'),
    re.compile(r'^\\begin\{flushleft\}$'),
    re.compile(r'^\\end\{flushleft\}$'),
    re.compile(r'^\\hfill\s*$'),
    re.compile(r'^\\vspace\{.*?\}\s*$'),
    re.compile(r'^\\newpage\s*$'),
    re.compile(r'^\\clearpage\s*$'),
    re.compile(r'^\\centering\s*$'),
    re.compile(r'^万方数据\s*$'),
    re.compile(r'^维普资讯\s*$'),
    re.compile(r'^中国知网\s*$'),
    re.compile(r'^CNKI\s*$', re.IGNORECASE),
    re.compile(r'^\d{1,5}\s*$'),
]


def _is_skip_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    return any(p.match(stripped) for p in _JUNK_LINE_PATTERNS)


def _segment_page(content: str) -> List[Tuple[str, str]]:
    """将单页内容分割为有序的 (type, text) 片段列表。

    type 为 'figure' | 'latex_env' | 'display_math' | 'text'。

    display_math 只匹配独占段落的 $$...$$ (跨行或前后为换行)。
    行内 $$...$$ 已在预处理中被 _normalize_inline_math 转为 $...$。
    """
    segments: List[Tuple[str, str]] = []

    combined = re.compile(
        r'(?P<figure>\\begin\{figure\}[^\n]*\n[\s\S]*?\\end\{figure\})'
        r'|(?P<latex_env>'
        r'\\begin\{(?:equation|align|gather|multline|eqnarray|cases|tabular|tabularx|table)[*]?\}'
        r'[\s\S]*?'
        r'\\end\{(?:equation|align|gather|multline|eqnarray|cases|tabular|tabularx|table)[*]?\})'
        r'|(?P<display_math>(?:^|\n)\s*\$\$[\s\S]+?\$\$\s*(?:\n|$))'
    )

    last_end = 0
    for m in combined.finditer(content):
        if m.start() > last_end:
            text_part = content[last_end:m.start()].strip()
            if text_part:
                segments.append(('text', text_part))

        if m.group('figure'):
            caption = _extract_balanced_caption(m.group())
            segments.append(('figure', caption))
        elif m.group('latex_env'):
            segments.append(('latex_env', m.group()))
        elif m.group('display_math'):
            segments.append(('display_math', m.group().strip()))

        last_end = m.end()

    if last_end < len(content):
        text_part = content[last_end:].strip()
        if text_part:
            segments.append(('text', text_part))

    return segments


def _text_to_chunks(text: str, page_num: int) -> List[Dict]:
    """将纯文本片段解析为 header / text_group chunk。"""
    chunks: List[Dict] = []
    lines = text.split('\n')
    current_lines: List[str] = []

    def flush():
        nonlocal current_lines
        if not current_lines:
            return
        clean = [l.strip() for l in current_lines if l.strip()]
        if clean:
            chunks.append({
                'type': 'text_group',
                'chunks': [{'page': page_num, 'type': 'text', 'content': clean}],
            })
        current_lines = []

    for line in lines:
        stripped = line.strip()
        if _is_skip_line(stripped):
            continue

        if _is_header_line(stripped):
            flush()
            chunks.append({
                'type': 'header',
                'page': page_num,
                'content': [_extract_header_text(stripped)],
            })
            continue

        current_lines.append(stripped)

    flush()
    return chunks


_LINE_NUMBER_ROW_RE = re.compile(r'^\s*\d{1,4}\s*(?:\\\\|&|$)')
_TABULAR_ENV_RE = re.compile(
    r'\\begin\{(?:tabular|tabularx)\}(?:\{[^}]*\})?\{([^}]*)\}',
)


_JOURNAL_META_KEYWORDS = re.compile(
    r'ISSN|Front(?:iers)?.*(?:Inform|Technol)|www\.\w+\.\w+|'
    r'E-mail:\s*\w|doi\.org|springerlink|CLC\s+number',
    re.IGNORECASE,
)


def _is_line_number_table(tex: str) -> bool:
    """判断 tabular 环境是否仅包含行号（OCR 误提取的侧边行号）或期刊元信息。"""
    body_match = re.search(
        r'\\begin\{(?:tabular|tabularx)\}.*?\n([\s\S]*?)\\end\{(?:tabular|tabularx)\}',
        tex,
    )
    if not body_match:
        return False
    body = body_match.group(1)
    if _JOURNAL_META_KEYWORDS.search(body):
        return True
    rows = [r.strip() for r in body.split('\\\\') if r.strip()]
    rows = [r for r in rows if r and r != '\\hline']
    if not rows:
        return True
    num_rows = sum(1 for r in rows if _LINE_NUMBER_ROW_RE.match(r))
    return len(rows) >= 3 and num_rows / len(rows) > 0.7


def _clean_line_number_columns(tex: str) -> str:
    """从混合内容的 tabular 中移除行号列，保留实际文本内容。

    对于含行号+正文的多列表格（如 ``{l p{4.5cm} p{4.5cm}}``），
    去掉第一列（行号），将其余列合并为纯文本段落。
    """
    col_match = _TABULAR_ENV_RE.search(tex)
    if not col_match:
        return tex
    cols = col_match.group(1)
    if cols.count('p') == 0 and cols.count('X') == 0:
        return tex

    body_match = re.search(
        r'\\begin\{(?:tabular|tabularx)\}.*?\n([\s\S]*?)\\end\{(?:tabular|tabularx)\}',
        tex,
    )
    if not body_match:
        return tex

    body = body_match.group(1)
    rows = body.split('\\\\')
    text_parts: List[str] = []
    for row in rows:
        row = row.strip()
        if not row or row == '\\hline':
            continue
        cells = row.split('&')
        first = cells[0].strip()
        if re.match(r'^\d{1,4}$', first):
            rest = ' '.join(c.strip() for c in cells[1:] if c.strip())
            if rest:
                text_parts.append(rest)
        else:
            text_parts.append(row)
    return '\n'.join(text_parts)


def _filter_segments(segments: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """过滤并清洗 segment 列表，移除纯行号表格，清理混合表格。"""
    result: List[Tuple[str, str]] = []
    for seg_type, seg_text in segments:
        if seg_type == 'latex_env':
            if _is_line_number_table(seg_text):
                cleaned = _clean_line_number_columns(seg_text)
                if cleaned.strip() and cleaned != seg_text:
                    result.append(('text', cleaned))
                continue
        result.append((seg_type, seg_text))
    return result


def parse_ocr_to_chunks(full_md: str) -> List[Dict]:
    """将 Qwen-VL-OCR 输出转换为 pipeline chunk 列表。"""
    pages = _split_pages(full_md)
    if not pages and (full_md or "").strip():
        body = _strip_code_fences(full_md)
        if body.strip():
            pages = [(1, body)]

    all_chunks: List[Dict] = []

    for page_num, raw_content in pages:
        content = _strip_code_fences(raw_content)
        content = _clean_page_junk(content)
        content = _normalize_inline_math(content)
        segments = _segment_page(content)
        segments = _filter_segments(segments)
        figure_slot = 0
        page_has_content = False

        for seg_type, seg_text in segments:
            if seg_type == 'figure':
                figure_slot += 1
                all_chunks.append({
                    'type': 'image',
                    'page': page_num,
                    'figure_slot': figure_slot,
                    'bbox': (0, 0, 0, 0),
                    'caption': seg_text,
                })
                page_has_content = True

            elif seg_type in ('latex_env', 'display_math'):
                all_chunks.append({
                    'type': 'latex_raw',
                    'page': page_num,
                    'content': seg_text,
                })
                page_has_content = True

            elif seg_type == 'text':
                sub_chunks = _text_to_chunks(seg_text, page_num)
                if sub_chunks:
                    all_chunks.extend(sub_chunks)
                    page_has_content = True

        if content.strip() and not page_has_content:
            all_chunks.append({
                'type': 'text_group',
                'chunks': [{'page': page_num, 'type': 'text',
                            'content': [content.strip()[:50000]]}],
            })

    return all_chunks
