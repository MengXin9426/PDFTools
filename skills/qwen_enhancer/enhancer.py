"""
LLM 增强器 — 统一封装 Qwen 云端 API 与本地 vLLM 服务。

翻译后端通过 config.yaml 的 ``translate_backend`` 字段切换（"qwen" / "vllm"），
也可通过构造函数参数 ``backend`` 显式指定。两种后端均使用 OpenAI 兼容接口。
"""

import base64
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

from ..load_api_config import (
    translate_backend as cfg_translate_backend,
    qwen_api_key,
    qwen_base_url,
    qwen_translate_model,
    vllm_api_key,
    vllm_base_url,
    vllm_model,
)


class QwenEnhancer:
    """Qwen / vLLM 统一增强器（翻译 + 图像理解）。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        self.backend = (backend or cfg_translate_backend()).lower()

        if self.backend == "vllm":
            self.api_key = (api_key or vllm_api_key()).strip()
            self.model = model or vllm_model()
            base_url = vllm_base_url()
        else:
            self.api_key = (api_key or qwen_api_key() or "").strip()
            self.model = model or qwen_translate_model()
            base_url = qwen_base_url()

        self.client: Optional[OpenAI] = None
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=base_url,
            )
        print(f"  ℹ LLM 后端: {self.backend} | model={self.model} | url={base_url}")

    def _ready(self) -> bool:
        return self.client is not None

    def _extra_create_kwargs(self) -> dict:
        """vLLM 后端需要关闭 thinking 模式，否则 content 为空。"""
        if self.backend == "vllm":
            return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        return {}

    # ------------------------------------------------------------------ #
    #  文本翻译
    # ------------------------------------------------------------------ #

    def enhance_text_translation(self, text: str, context: Optional[str] = None) -> str:
        """学术英文 → 中文翻译。"""
        if not self._ready():
            print("  ⚠ 未配置 API Key，跳过翻译")
            return text
        try:
            ctx = f"\n上下文：{context}" if context else ""
            prompt = (
                "请将以下学术文本翻译成中文，要求：\n"
                "1. 保持专业术语的准确性\n"
                "2. 符合中文学术表达习惯\n"
                "3. 专有名词、缩写、型号（如 DSSS、OFDM、MIMO、IEEE 等）保持原样不翻译\n"
                "4. 数学公式、环境（equation/align 等）与 LaTeX 命令保持原样；占位符行不得改写\n"
                "5. 变量名、函数名、单位符号保持英文或原样；保持与原文一致的换行结构\n\n"
                f"待翻译文本：\n{text}{ctx}\n\n"
                "请只返回翻译结果，不要其他说明。"
            )
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一个专业的学术翻译助手，擅长翻译英文科技论文，"
                            "特别关注通信、信号处理和雷达技术领域。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=4096,
                **self._extra_create_kwargs(),
            )
            msg = resp.choices[0].message
            raw = getattr(msg, "content", None)
            if raw is None:
                print("  ⚠ Qwen 返回 message.content 为 None，保留原文")
                return text
            if not isinstance(raw, str):
                raw = str(raw)
            raw = raw.strip()
            if not raw:
                print("  ⚠ Qwen 返回空字符串，保留原文")
                return text
            return raw
        except Exception as e:
            print(f"  ⚠ Qwen 翻译失败: {e}")
            return text

    # ------------------------------------------------------------------ #
    #  图片理解
    # ------------------------------------------------------------------ #

    def enhance_image_understanding(self, image_path: str, prompt: Optional[str] = None) -> str:
        if not self._ready():
            print("  ⚠ 未配置 QWEN_API_KEY，跳过 Qwen 图片理解")
            return ""
        try:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            default_prompt = (
                "请详细分析这张图片的内容，包括：\n"
                "1. 图片类型（表格、图表、公式、插图等）\n"
                "2. 主要内容和结构\n"
                "3. 文字信息（如果有）\n"
                "4. 数值数据（如果有）\n"
                "5. 图表说明和注释\n\n"
                "请用中文回答，结构清晰。"
            )
            resp = self.client.chat.completions.create(
                model="qwen-vl-plus",
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的 PDF 内容分析助手，擅长识别和分析学术论文中的图片、表格和公式。",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": prompt or default_prompt},
                        ],
                    },
                ],
                temperature=0.3,
                max_tokens=4096,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠ Qwen 图片理解失败: {e}")
            return ""

    def enhance_math_extraction(self, image_path: str) -> str:
        prompt = (
            "请识别这张图片中的数学表达式或公式，并以 LaTeX 格式返回。\n"
            "要求：\n"
            "1. 保持原有的数学结构\n"
            "2. 使用标准的 LaTeX 命令\n"
            "3. 只返回 LaTeX 代码，不要其他说明"
        )
        return self.enhance_image_understanding(image_path, prompt)

    def enhance_table_extraction(self, image_path: str) -> str:
        prompt = (
            "请识别这张图片中的表格内容，并以 Markdown 表格格式返回。\n"
            "要求：保持行列结构，准确提取每个单元格文字。请用中文回答。"
        )
        return self.enhance_image_understanding(image_path, prompt)

    def enhance_chart_analysis(self, image_path: str) -> str:
        prompt = (
            "请详细分析这张图表，包括：图表类型、X/Y 轴含义、"
            "数据趋势、关键数据点和主要结论。请用中文回答。"
        )
        return self.enhance_image_understanding(image_path, prompt)

    def enhance_figure_caption(self, image_path: str, text: str) -> Dict:
        if not self._ready():
            return {"type": "unknown", "description": "", "caption_cn": text, "key_info": []}
        try:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            prompt = (
                f"基于这张图片和原始文本，请生成详细的图片说明。\n\n"
                f"原始文本：{text}\n\n"
                "请用 JSON 格式返回：\n"
                '{"type":"图表类型","description":"内容描述","caption_cn":"中文标题","key_info":["关键信息1"]}'
            )
            resp = self.client.chat.completions.create(
                model="qwen-vl-plus",
                messages=[
                    {"role": "system", "content": "你是一个专业的学术论文图片分析助手。"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            import json

            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.endswith("```"):
                raw = raw[:-3]
            return json.loads(raw.strip())
        except Exception:
            return {"type": "unknown", "description": "", "caption_cn": text, "key_info": []}

    # ------------------------------------------------------------------ #
    #  批量处理 & fallback
    # ------------------------------------------------------------------ #

    def batch_enhance_images(self, image_dir: Path, output_dir: Optional[Path] = None) -> List[Dict]:
        results: List[Dict] = []
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        image_files = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
        print(f"  → 找到 {len(image_files)} 张图片，开始 Qwen 增强...")
        for i, img in enumerate(image_files, 1):
            print(f"  → 处理 {i}/{len(image_files)}: {img.name}")
            try:
                if "tables" in str(img):
                    analysis, img_type = self.enhance_table_extraction(str(img)), "table"
                elif "figures" in str(img):
                    analysis, img_type = self.enhance_chart_analysis(str(img)), "figure"
                elif "equations" in str(img):
                    analysis, img_type = self.enhance_math_extraction(str(img)), "equation"
                else:
                    analysis, img_type = self.enhance_image_understanding(str(img)), "unknown"
                result = {"filename": img.name, "type": img_type, "analysis": analysis, "enhanced": True}
                results.append(result)
                if output_dir:
                    (output_dir / f"{img.stem}_analysis.txt").write_text(
                        f"图片文件: {img.name}\n类型: {img_type}\n\n分析结果:\n{analysis}\n",
                        encoding="utf-8",
                    )
                print(f"  ✓ 完成: {img.name}")
            except Exception as e:
                print(f"  ✗ 失败: {img.name} - {e}")
                results.append({"filename": img.name, "type": "unknown", "analysis": "", "enhanced": False, "error": str(e)})
        print(f"\n  ✓ Qwen 增强完成: {len(results)} 张图片")
        return results

    def fallback_translation(self, text: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                result = self.enhance_text_translation(text)
                if result and result != text:
                    return result
            except Exception as e:
                print(f"  ⚠ 翻译重试 {attempt + 1}/{max_retries}: {e}")
        return f"[翻译失败，保留原文]{text}"
