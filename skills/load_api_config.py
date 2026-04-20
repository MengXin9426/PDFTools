"""统一配置加载器 — 从环境变量或项目根目录 ``config.yaml`` 读取配置。

优先级：环境变量 > config.yaml > 默认值
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_config_cache: Optional[Dict[str, Any]] = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config() -> Dict[str, Any]:
    """加载并缓存 config.yaml 配置。"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    cfg_path = _repo_root() / "config.yaml"
    if cfg_path.is_file():
        try:
            _config_cache = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            _config_cache = {}
    else:
        _config_cache = {}
    return _config_cache


# ------------------------------------------------------------------ #
#  翻译后端选择
# ------------------------------------------------------------------ #

def translate_backend() -> str:
    """当前翻译后端：``TRANSLATE_BACKEND`` 环境变量 > ``config.yaml`` > 默认 ``qwen``。

    合法值: "qwen" | "vllm"
    """
    env = (os.environ.get("TRANSLATE_BACKEND") or "").strip().lower()
    if env in ("qwen", "vllm"):
        return env
    cfg = load_config()
    val = str(cfg.get("translate_backend") or "qwen").strip().lower()
    return val if val in ("qwen", "vllm") else "qwen"


# ------------------------------------------------------------------ #
#  Qwen (阿里云 DashScope) 配置
# ------------------------------------------------------------------ #

def qwen_api_key() -> Optional[str]:
    k = (os.environ.get("QWEN_API_KEY") or "").strip()
    if k:
        return k
    cfg = load_config()
    k = str((cfg.get("api") or {}).get("qwen", {}).get("api_key") or "").strip()
    return k or None


def qwen_base_url() -> str:
    cfg = load_config()
    url = (cfg.get("api") or {}).get("qwen", {}).get("base_url") or ""
    return url.strip() or "https://dashscope.aliyuncs.com/compatible-mode/v1"


def qwen_ocr_model() -> str:
    cfg = load_config()
    m = (cfg.get("api") or {}).get("qwen", {}).get("ocr_model") or ""
    return m.strip() or "qwen-vl-ocr"


def qwen_translate_model() -> str:
    cfg = load_config()
    m = (cfg.get("api") or {}).get("qwen", {}).get("translate_model") or ""
    return m.strip() or "qwen-plus"


# ------------------------------------------------------------------ #
#  vLLM 本地服务配置
# ------------------------------------------------------------------ #

def vllm_api_key() -> str:
    """vLLM API Key：``VLLM_API_KEY`` 环境变量 > ``config.yaml`` > ``EMPTY``。"""
    k = (os.environ.get("VLLM_API_KEY") or "").strip()
    if k:
        return k
    cfg = load_config()
    k = str((cfg.get("api") or {}).get("vllm", {}).get("api_key") or "").strip()
    return k or "EMPTY"


def vllm_base_url() -> str:
    cfg = load_config()
    url = (cfg.get("api") or {}).get("vllm", {}).get("base_url") or ""
    return url.strip() or "http://localhost:8000/v1"


def vllm_model() -> str:
    cfg = load_config()
    m = (cfg.get("api") or {}).get("vllm", {}).get("model") or ""
    return m.strip() or "Qwen/Qwen2.5-7B-Instruct"


def vllm_timeout() -> int:
    cfg = load_config()
    t = (cfg.get("api") or {}).get("vllm", {}).get("timeout")
    return int(t) if t else 120


# ------------------------------------------------------------------ #
#  根据 backend 自动选择翻译用的 api_key / base_url / model
# ------------------------------------------------------------------ #

def active_translate_api_key() -> str:
    """根据当前 backend 返回对应 api_key。"""
    if translate_backend() == "vllm":
        return vllm_api_key()
    return qwen_api_key() or ""


def active_translate_base_url() -> str:
    if translate_backend() == "vllm":
        return vllm_base_url()
    return qwen_base_url()


def active_translate_model() -> str:
    if translate_backend() == "vllm":
        return vllm_model()
    return qwen_translate_model()


# ------------------------------------------------------------------ #
#  版面检测
# ------------------------------------------------------------------ #

def layout_detector_config() -> Dict[str, Any]:
    cfg = load_config()
    ld = cfg.get("layout_detector") or {}
    return {
        "local_path": str(ld.get("local_path") or "").strip(),
        "model_repo": ld.get("model_repo") or "juliozhao/DocLayout-YOLO-DocStructBench",
        "model_file": ld.get("model_file") or "doclayout_yolo_docstructbench_imgsz1024.pt",
        "weights_dir": ld.get("weights_dir") or "models",
        "hf_mirror": ld.get("hf_mirror") or "https://hf-mirror.com",
        "device": ld.get("device") or "cpu",
        "confidence": float(ld.get("confidence") or 0.3),
    }


# ------------------------------------------------------------------ #
#  LaTeX / 排版
# ------------------------------------------------------------------ #

def latex_config() -> Dict[str, Any]:
    cfg = load_config()
    lx = cfg.get("latex") or {}
    layout = str(lx.get("layout") or "auto").strip().lower()
    if layout not in ("twocolumn", "onecolumn", "auto"):
        layout = "auto"
    return {
        "compiler": lx.get("compiler") or "xelatex",
        "timeout": int(lx.get("timeout") or 300),
        "layout": layout,
        "auto_layout_severe_threshold": int(lx.get("auto_layout_severe_threshold") or 80),
        "auto_layout_max_severe": int(lx.get("auto_layout_max_severe") or 15),
    }
