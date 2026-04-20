"""LaTeX 文档构建器 — 将翻译后的 chunk + 图片重组为学术 PDF。

核心原则：
    - equation/align/tabular 等 LaTeX 环境原样写入，不做字符转义
    - 翻译后的正文：含 LaTeX 命令的行原样输出，纯文本行做最小转义
    - figure 环境用裁剪的图表图或页面渲染图替换
    - 自适应排版：先尝试双栏，溢出严重则自动切换单栏
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re


class LatexBuilder:

    def __init__(self, output_dir: str = "output", layout: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.explicit_translated_pdf: Optional[Path] = None

        from ..load_api_config import latex_config
        lx = latex_config()
        self._layout = (layout or lx["layout"]).lower()
        self._severe_threshold = lx["auto_layout_severe_threshold"]
        self._max_severe = lx["auto_layout_max_severe"]

    def build_latex(
        self,
        pdf_name: str,
        chunks: List[Dict],
        images: List[Dict],
        page_images_dir: Optional[str] = None,
    ) -> Optional[str]:
        self.explicit_translated_pdf = None
        print(f"\n构建LaTeX文档: {pdf_name}（排版模式: {self._layout}）")

        latex_dir = self.output_dir / "result" / pdf_name
        latex_dir.mkdir(parents=True, exist_ok=True)

        if page_images_dir:
            self._copy_page_images_to_result(page_images_dir, latex_dir)

        latex_path = latex_dir / "document.tex"

        if self._layout == "onecolumn":
            pdf_path = self._build_single(latex_path, pdf_name, chunks, images, page_images_dir, twocolumn=False)
        elif self._layout == "twocolumn":
            pdf_path = self._build_single(latex_path, pdf_name, chunks, images, page_images_dir, twocolumn=True)
        else:
            pdf_path = self._build_auto(latex_path, pdf_name, chunks, images, page_images_dir)

        if not pdf_path:
            print(f"  ℹ 未生成 PDF，可稍后使用 xelatex 编译: {latex_path}")
            return None

        pdf_path = Path(pdf_path)
        alias = (self.output_dir / f"{pdf_name}_translated.pdf").resolve()
        try:
            shutil.copy2(pdf_path, alias)
            self.explicit_translated_pdf = alias
            inner = (latex_dir / "translated_final.pdf").resolve()
            shutil.copy2(pdf_path, inner)
            print(f"  ✓ 翻译 PDF（固定路径）: {alias}")
            print(f"  ✓ 结果目录副本:        {inner}")
        except OSError as e:
            print(f"  ⚠ 复制翻译 PDF 失败: {e}")

        return str(pdf_path)

    def _build_single(self, latex_path, pdf_name, chunks, images, page_images_dir, twocolumn):
        mode_str = "双栏" if twocolumn else "单栏"
        content = self._generate_latex_content(pdf_name, chunks, images, page_images_dir, twocolumn=twocolumn)
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ LaTeX文件已保存（{mode_str}）: {latex_path}")
        pdf = self._compile_latex(latex_path)
        if pdf:
            severe = self._count_severe_overfull(latex_path.with_suffix(".log"))
            print(f"  ℹ {mode_str}编译：严重溢出 {severe} 处（>{self._severe_threshold}pt）")
        return pdf

    def _build_auto(self, latex_path, pdf_name, chunks, images, page_images_dir):
        twocol_content = self._generate_latex_content(pdf_name, chunks, images, page_images_dir, twocolumn=True)
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(twocol_content)
        print(f"  ✓ LaTeX文件已保存（双栏/auto）: {latex_path}")

        pdf_path = self._compile_latex(latex_path)
        if not pdf_path:
            return None

        severe = self._count_severe_overfull(latex_path.with_suffix(".log"))
        print(f"  ℹ 双栏编译：严重溢出 {severe} 处（阈值 >{self._severe_threshold}pt）")

        if severe > self._max_severe:
            print(f"  → 严重溢出超限（{severe}>{self._max_severe}），切换单栏重编...")
            onecol_content = self._generate_latex_content(pdf_name, chunks, images, page_images_dir, twocolumn=False)
            with open(latex_path, 'w', encoding='utf-8') as f:
                f.write(onecol_content)
            pdf_path_2 = self._compile_latex(latex_path)
            if pdf_path_2:
                severe2 = self._count_severe_overfull(latex_path.with_suffix(".log"))
                print(f"  ✓ 单栏编译完成，严重溢出 {severe2} 处")
                return pdf_path_2
            else:
                print("  ⚠ 单栏编译失败，保留双栏结果")
                with open(latex_path, 'w', encoding='utf-8') as f:
                    f.write(twocol_content)
                self._compile_latex(latex_path)
                return pdf_path
        else:
            print("  ✓ 双栏排版良好，无需切换")
            return pdf_path

    def _copy_page_images_to_result(self, page_images_dir: str, latex_dir: Path) -> None:
        src = Path(page_images_dir)
        if not src.is_dir():
            return
        dest = latex_dir / "page_images"
        dest.mkdir(exist_ok=True)
        for f in sorted(src.glob("*.png")):
            shutil.copy2(f, dest / f.name)

    _RE_OVERFULL_PT = re.compile(r'Overfull \\hbox \((\d+(?:\.\d+)?)pt too wide\)')

    def _count_severe_overfull(self, log_path: Path) -> int:
        if not log_path.is_file():
            return 0
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return 0
        count = 0
        for m in self._RE_OVERFULL_PT.finditer(text):
            if float(m.group(1)) > self._severe_threshold:
                count += 1
        return count

    # ------------------------------------------------------------------ #
    #  LaTeX 内容生成
    # ------------------------------------------------------------------ #

    _RE_REF_HEADING = re.compile(
        r'^(参考文献|references|bibliography)$', re.IGNORECASE,
    )
    _RE_TABULAR_BEGIN = re.compile(
        r'\\begin\{tabular\}\{([^}]*)\}',
    )

    def _generate_latex_content(
        self,
        title: str,
        chunks: List[Dict],
        images: List[Dict],
        page_images_dir: Optional[str] = None,
        twocolumn: bool = True,
    ) -> str:
        safe_title = self._escape_text(title)

        if twocolumn:
            doc_class = r"\documentclass[10pt,a4paper,twocolumn]{article}"
            col_settings = (
                r"\setlength{\columnsep}{5mm}" "\n"
            )
        else:
            doc_class = r"\documentclass[11pt,a4paper]{article}"
            col_settings = ""

        preamble = (
            doc_class + "\n"
            r"\usepackage{fontspec}" "\n"
            r"\setmainfont{Noto Sans CJK SC}" "\n"
            r"\usepackage{graphicx}" "\n"
            r"\usepackage{float}" "\n"
            r"\usepackage{amsmath,amssymb,amsfonts}" "\n"
            r"\usepackage{bm}" "\n"
            r"\usepackage{mathtools}" "\n"
            r"\usepackage[margin=1.8cm]{geometry}" "\n"
            r"\usepackage{hyperref}" "\n"
            r"\hypersetup{unicode=true,colorlinks=true,linkcolor=blue,citecolor=blue,"
            r"urlcolor=blue,breaklinks=true}" "\n"
            r"\usepackage{booktabs}" "\n"
            r"\usepackage{caption}" "\n"
            r"\captionsetup{font=small,labelfont=bf}" "\n"
            r"\usepackage{tabularx}" "\n"
            r"\usepackage{microtype}" "\n"
            r"\usepackage{url}" "\n"
            r"\Urlmuskip=0mu plus 10mu" "\n"
            "\n"
            r"\sloppy" "\n"
            r"\tolerance=3000" "\n"
            r"\emergencystretch=5em" "\n"
            + col_settings +
            r"\setlength{\parindent}{2em}" "\n"
            r"\setlength{\parskip}{0.3em}" "\n"
            "\n"
        )

        first_section_idx = self._find_first_section_idx(chunks)
        header_chunks = chunks[:first_section_idx]
        body_chunks = chunks[first_section_idx:]

        header_tex = self._render_header_area(
            safe_title, header_chunks, images, page_images_dir, twocolumn,
        )

        body_parts: List[str] = []
        in_references = False

        for chunk in body_chunks:
            ctype = chunk.get('type', '')

            if ctype == 'header':
                hdr_text = (chunk.get('content') or [''])[0]
                hdr = self._escape_text(hdr_text)
                if self._RE_REF_HEADING.search(hdr_text.strip()):
                    in_references = True
                    body_parts.append(f"\\section*{{{hdr}}}\n")
                    body_parts.append("\\begin{flushleft}\n\\small\n")
                else:
                    if in_references:
                        body_parts.append("\\end{flushleft}\n")
                        in_references = False
                    body_parts.append(f"\\section*{{{hdr}}}\n")

            elif ctype == 'latex_raw':
                raw = self._balance_braces(chunk.get('content', ''))
                raw = self._strip_flushright(raw)
                raw = self._fix_tabular_width(raw)
                if raw.strip():
                    body_parts.append(f"\n{raw}\n")

            elif ctype == 'image':
                body_parts.append(
                    self._render_figure(chunk, images, page_images_dir, twocolumn)
                )

            elif ctype == 'text_group':
                for tc in chunk.get('chunks', []):
                    for line in tc.get('content', []):
                        body_parts.append(self._format_text_line(line) + "\n")
                    body_parts.append("\n")

        if in_references:
            body_parts.append("\\end{flushleft}\n")

        body_parts.append("\\end{document}\n")
        return preamble + header_tex + "".join(body_parts)

    # ------------------------------------------------------------------ #
    #  Header / Section detection
    # ------------------------------------------------------------------ #

    _RE_NUMBERED_SECTION = re.compile(r'^\d+[\.\s]')
    _MAX_HEADER_CHUNKS = 6

    @staticmethod
    def _find_first_section_idx(chunks: List[Dict]) -> int:
        _env_re = re.compile(
            r'\\begin\{(tabular|tabularx|table|figure|algorithm|lstlisting)\}',
        )
        for i, c in enumerate(chunks):
            if i >= LatexBuilder._MAX_HEADER_CHUNKS:
                return i
            ctype = c.get('type', '')
            if ctype == 'header':
                content = (c.get('content') or [''])[0].strip()
                if LatexBuilder._RE_NUMBERED_SECTION.match(content):
                    return i
            elif ctype == 'latex_raw':
                raw = c.get('content', '')
                if _env_re.search(raw):
                    return i
        return min(len(chunks), LatexBuilder._MAX_HEADER_CHUNKS)

    # ------------------------------------------------------------------ #
    #  Header area rendering
    # ------------------------------------------------------------------ #

    _RE_ABSTRACT_ENV = re.compile(r'\\begin\{abstract\}|\\end\{abstract\}')
    _RE_FLUSHRIGHT = re.compile(
        r'\\begin\{flushright\}[\s\S]*?\\end\{flushright\}',
    )

    @classmethod
    def _strip_flushright(cls, tex: str) -> str:
        return cls._RE_FLUSHRIGHT.sub('', tex)

    def _render_header_area(
        self,
        safe_title: str,
        header_chunks: List[Dict],
        images: List[Dict],
        page_images_dir: Optional[str],
        twocolumn: bool = True,
    ) -> str:
        meta_lines: List[str] = []
        deferred_figures: List[str] = []

        for chunk in header_chunks:
            ctype = chunk.get('type', '')
            if ctype == 'text_group':
                for tc in chunk.get('chunks', []):
                    for line in tc.get('content', []):
                        meta_lines.append(self._format_text_line(line))
            elif ctype == 'latex_raw':
                raw = self._balance_braces(chunk.get('content', ''))
                raw = self._strip_flushright(raw)
                raw = self._fix_tabular_width(raw)
                raw = self._RE_ABSTRACT_ENV.sub('', raw)
                if raw.strip():
                    meta_lines.append(raw)
            elif ctype == 'image':
                deferred_figures.append(
                    self._render_figure(chunk, images, page_images_dir, twocolumn)
                )

        meta_text = "\n".join(meta_lines).strip()

        if not twocolumn:
            result = (
                f"\\begin{{document}}\n"
                f"\\begin{{center}}\n"
                f"  {{\\LARGE\\bfseries {safe_title}}}\n"
                f"  \\vspace{{1em}}\n"
                f"\\end{{center}}\n"
            )
            if meta_text:
                result += f"{meta_text}\n\\vspace{{1em}}\n\n"
            else:
                result += "\n"
        elif not meta_text:
            result = (
                f"\\begin{{document}}\n"
                f"\\twocolumn[\n"
                f"  \\begin{{center}}\n"
                f"    {{\\LARGE\\bfseries {safe_title}}}\n"
                f"    \\vspace{{1em}}\n"
                f"  \\end{{center}}\n"
                f"]\n\n"
            )
        elif len(meta_text) > 2000:
            result = (
                f"\\begin{{document}}\n"
                f"\\twocolumn[\n"
                f"  \\begin{{center}}\n"
                f"    {{\\LARGE\\bfseries {safe_title}}}\n"
                f"    \\vspace{{0.5em}}\n"
                f"  \\end{{center}}\n"
                f"]\n\n"
                f"{meta_text}\n\n"
            )
        else:
            result = (
                f"\\begin{{document}}\n"
                f"\\twocolumn[\n"
                f"  \\begin{{center}}\n"
                f"    {{\\LARGE\\bfseries {safe_title}}}\n"
                f"    \\vspace{{0.8em}}\n"
                f"  \\end{{center}}\n"
                f"  \\begin{{minipage}}{{\\textwidth}}\n"
                f"    \\small\n"
                f"    {meta_text}\n"
                f"  \\end{{minipage}}\n"
                f"  \\vspace{{1em}}\n"
                f"]\n\n"
            )

        for fig in deferred_figures:
            result += fig
        return result

    # ------------------------------------------------------------------ #
    #  Figure rendering
    # ------------------------------------------------------------------ #

    def _render_figure(
        self,
        chunk: Dict,
        images: List[Dict],
        page_images_dir: Optional[str],
        twocolumn: bool = True,
    ) -> str:
        img = self._pick_image_for_chunk(images, chunk)
        cap_raw = chunk.get('caption', '') or 'Figure'
        cap = self._balance_braces(cap_raw)

        width = "0.95\\linewidth" if twocolumn else "0.7\\textwidth"
        parts: List[str] = []
        parts.append("\\begin{figure}[htbp]\n  \\centering\n")
        if img:
            rel = f"images/{img['filename']}".replace("\\", "/")
            parts.append(
                f"  \\includegraphics[width={width},"
                f"keepaspectratio]{{{rel}}}\n"
            )
        elif page_images_dir:
            page_num = chunk.get("page", 0)
            parts.append(
                f"  \\includegraphics[width={width},"
                f"keepaspectratio]"
                f"{{page_images/page_{page_num:03d}.png}}\n"
            )
        else:
            parts.append("  % 未找到对应图片\n")
        parts.append(f"  \\caption{{{cap}}}\n\\end{{figure}}\n")
        return "".join(parts)

    # ------------------------------------------------------------------ #
    #  Tabular fixes
    # ------------------------------------------------------------------ #

    @classmethod
    def _fix_tabular_width(cls, tex: str) -> str:
        def _replace(m: re.Match) -> str:
            cols = m.group(1)
            x_cols = re.sub(r'l', 'X', cols, count=1)
            return f"\\begin{{tabularx}}{{\\linewidth}}{{{x_cols}}}"
        result = cls._RE_TABULAR_BEGIN.sub(_replace, tex)
        result = result.replace(r'\end{tabular}', r'\end{tabularx}')
        result = cls._fix_column_count(result)
        return result

    @staticmethod
    def _fix_column_count(tex: str) -> str:
        env_re = re.compile(
            r'(\\begin\{(?:tabularx|tabular)\}(?:\{[^}]*\})?\{)([^}]*?)(\})'
            r'([\s\S]*?)'
            r'(\\end\{(?:tabularx|tabular)\})',
        )

        def _fix(m: re.Match) -> str:
            prefix, col_spec, brace, body, end = m.groups()
            defined_cols = len(re.findall(r'[lcrXp]', col_spec))
            rows = body.split('\\\\')
            max_amps = 0
            for row in rows:
                row_clean = row.strip()
                if not row_clean or row_clean == '\\hline':
                    continue
                max_amps = max(max_amps, row_clean.count('&'))
            needed_cols = max_amps + 1
            if needed_cols > defined_cols:
                col_spec = col_spec + 'c' * (needed_cols - defined_cols)
            return f"{prefix}{col_spec}{brace}{body}{end}"

        return env_re.sub(_fix, tex)

    # ------------------------------------------------------------------ #
    #  Text escaping
    # ------------------------------------------------------------------ #

    _RE_BARE_AMP = re.compile(r'(?<!\\)&')
    _RE_BARE_HASH = re.compile(r'(?<!\\)#')
    _RE_BARE_PCT = re.compile(r'(?<!\\)%')
    _RE_BARE_LT = re.compile(r'(?<!\\)<')
    _RE_BARE_GT = re.compile(r'(?<!\\)>')

    @staticmethod
    def _safe_escape_special(text: str) -> str:
        if not text:
            return text
        parts = re.split(r'(\$[^$]+?\$)', text)
        out = []
        for i, part in enumerate(parts):
            if i % 2 == 1:
                out.append(part)
            else:
                part = LatexBuilder._RE_BARE_AMP.sub(r'\\&', part)
                part = LatexBuilder._RE_BARE_HASH.sub(r'\\#', part)
                part = LatexBuilder._RE_BARE_PCT.sub(r'\\%', part)
                part = LatexBuilder._RE_BARE_LT.sub(r'\\textless{}', part)
                part = LatexBuilder._RE_BARE_GT.sub(r'\\textgreater{}', part)
                out.append(part)
        return ''.join(out)

    @staticmethod
    def _escape_text(text: str) -> str:
        if not text:
            return text
        return LatexBuilder._safe_escape_special(text)

    @staticmethod
    def _balance_braces(text: str) -> str:
        if not text:
            return text
        depth = 0
        for ch in text:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
        if depth > 0:
            text += '}' * depth
        elif depth < 0:
            text = '{' * (-depth) + text
        return text

    _RE_PLACEHOLDER = re.compile(r'<<[A-Z_]+\d*>>')

    @staticmethod
    def _format_text_line(text: str) -> str:
        if not text:
            return text
        s = text.strip()
        if not s:
            return text
        s = LatexBuilder._RE_PLACEHOLDER.sub('', s)
        if not s.strip():
            return ''
        return LatexBuilder._safe_escape_special(s)

    @staticmethod
    def _pick_image_for_chunk(images: List[Dict], chunk: Dict) -> Optional[Dict]:
        page = chunk.get("page")
        slot = chunk.get("figure_slot") or 1
        on_page = sorted(
            (img for img in images if img.get("page") == page),
            key=lambda x: x.get("index", 0),
        )
        if not on_page:
            return None
        idx = min(max(int(slot), 1), len(on_page)) - 1
        return on_page[idx]

    # ------------------------------------------------------------------ #
    #  LaTeX compilation
    # ------------------------------------------------------------------ #

    def _compile_latex(self, latex_path: Path) -> Optional[Path]:
        from ..load_api_config import latex_config
        _lx = latex_config()
        timeout = int(os.environ.get("PDFTOOLS_LATEX_TIMEOUT", str(_lx["timeout"])))
        pdf_path = latex_path.with_suffix(".pdf")
        workdir = latex_path.parent
        tex_job = latex_path.name

        if pdf_path.exists():
            pdf_path.unlink()

        def _tail_build_log() -> None:
            log_file = latex_path.with_suffix(".log")
            if not log_file.is_file():
                return
            try:
                lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
                tail = "\n".join(lines[-35:])
                print(f"  ℹ {log_file.name}（末尾 35 行）:\n{tail}")
            except OSError:
                pass

        def _try_xelatex(passes: int = 2) -> bool:
            if not shutil.which("xelatex"):
                return False
            for _ in range(passes):
                subprocess.run(
                    ["xelatex", "-interaction=nonstopmode", tex_job],
                    cwd=workdir, capture_output=True, timeout=timeout,
                )
            return pdf_path.exists()

        try:
            if _try_xelatex():
                print(f"  ✓ PDF已生成 (xelatex): {pdf_path}")
                return pdf_path
            print("  ⚠ xelatex 未得到 PDF")
            _tail_build_log()
            return None
        except FileNotFoundError:
            print("  ⚠ 未找到 xelatex 编译器")
            return None
        except subprocess.TimeoutExpired:
            print(f"  ⚠ LaTeX 编译超时（{timeout}s）")
            return None
        except Exception as e:
            print(f"  ⚠ LaTeX编译错误: {e}")
            return None
