"""将翻译后的 chunk 列表导出为纯文本，便于核对翻译结果。"""

from pathlib import Path
from typing import List, Dict


def write_translated_plaintext(chunks: List[Dict], out_path) -> None:
    """把标题、正文（含括号内引文等原样随句翻译结果）写入 UTF-8 文本文件。"""
    out_path = Path(out_path)
    lines: List[str] = []
    for chunk in chunks:
        if chunk["type"] == "header":
            lines.append("")
            hdr = (chunk.get("content") or [""])[0]
            lines.append("## " + ("" if hdr is None else str(hdr)))
            lines.append("")
        elif chunk["type"] == "text_group":
            for tc in chunk.get("chunks", []):
                for line in tc.get("content", []):
                    lines.append("" if line is None else str(line))
            lines.append("")
        elif chunk["type"] == "latex_raw":
            lines.append("")
            lines.append(chunk.get("content", ""))
            lines.append("")
        elif chunk["type"] == "image":
            lines.append("")
            lines.append(f"[插图占位 第{chunk.get('page', '?')}页]")
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
