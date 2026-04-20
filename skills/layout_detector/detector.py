"""DocLayout-YOLO 版面检测器 — 从页面渲染图中检测并裁剪 figure/table 区域。

类别映射 (DocStructBench):
    0: title, 1: plain text, 2: abandon, 3: figure, 4: figure_caption,
    5: table, 6: table_caption, 7: table_footnote,
    8: isolate_formula, 9: formula_caption
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from ..load_api_config import layout_detector_config

logger = logging.getLogger(__name__)

_FIGURE_CLASSES = {"figure"}
_TABLE_CLASSES = {"table"}
_CAPTION_CLASSES = {"figure_caption", "table_caption"}
_EXTRACT_CLASSES = _FIGURE_CLASSES | _TABLE_CLASSES


def _iou(a: List[int], b: List[int]) -> float:
    """计算两个 bbox 的 IoU。"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _deduplicate_boxes(regions: List[Dict], iou_thresh: float = 0.5) -> List[Dict]:
    """按置信度降序去重，IoU 超过阈值的低置信度框被移除。"""
    by_conf = sorted(regions, key=lambda r: r["confidence"], reverse=True)
    keep: List[Dict] = []
    for r in by_conf:
        if any(_iou(r["bbox"], k["bbox"]) > iou_thresh for k in keep):
            continue
        keep.append(r)
    keep.sort(key=lambda r: r["bbox"][1])
    return keep


def _resolve_model_path(model_path: Optional[str] = None) -> str:
    """解析模型权重路径。

    优先级：构造函数参数 > config.yaml local_path > weights_dir 下搜索 > HF 下载
    """
    ld_cfg = layout_detector_config()
    model_file = ld_cfg["model_file"]

    if model_path and Path(model_path).is_file():
        return model_path

    local_path = ld_cfg.get("local_path", "")
    if local_path and Path(local_path).is_file():
        return local_path

    cache_root = Path(ld_cfg["weights_dir"])
    local_candidates = sorted(cache_root.rglob(model_file))
    if local_candidates:
        return str(local_candidates[0])

    try:
        import os
        if not os.environ.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = ld_cfg["hf_mirror"]
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=ld_cfg["model_repo"],
            filename=model_file,
            cache_dir=str(cache_root),
        )
    except Exception as e:
        raise FileNotFoundError(
            f"无法获取 DocLayout-YOLO 模型权重: {e}\n"
            f"请手动下载到 {cache_root}/{model_file}"
        ) from e


class LayoutDetector:
    """DocLayout-YOLO 版面检测器。"""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self._model = None
        self._model_path = model_path
        ld_cfg = layout_detector_config()
        self._device = device or ld_cfg["device"]

    def _ensure_model(self):
        if self._model is not None:
            return
        warnings.filterwarnings("ignore")
        from doclayout_yolo import YOLOv10
        resolved = _resolve_model_path(self._model_path)
        logger.info(f"加载 DocLayout-YOLO: {resolved}")
        self._model = YOLOv10(resolved)

    def detect_page(
        self,
        image_path: str,
        conf: float = 0.25,
        imgsz: int = 1024,
    ) -> List[Dict]:
        """检测单页中的所有区域，返回 bbox 列表。"""
        self._ensure_model()
        results = self._model.predict(
            image_path, imgsz=imgsz, conf=conf,
            device=self._device, verbose=False,
        )
        regions: List[Dict] = []
        for box in results[0].boxes:
            cls_id = int(box.cls)
            cls_name = results[0].names[cls_id]
            regions.append({
                "type": cls_name,
                "bbox": [int(x) for x in box.xyxy[0].tolist()],
                "confidence": round(float(box.conf), 3),
            })
        return regions

    def detect_and_crop_figures(
        self,
        page_images_dir: str,
        output_dir: str,
        conf: float = 0.25,
    ) -> List[Dict]:
        """对所有页面图片做版面检测，裁剪 figure/table 区域并保存。

        Returns:
            裁剪出的图片信息列表，格式：
            [{"page": int, "index": int, "type": str, "filename": str,
              "path": str, "bbox": list, "size": tuple}, ...]
        """
        src = Path(page_images_dir)
        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)

        page_files = sorted(src.glob("page_*.png"))
        if not page_files:
            logger.warning(f"未找到页面图片: {src}")
            return []

        all_crops: List[Dict] = []
        print(f"\n版面检测（DocLayout-YOLO）: {len(page_files)} 页")

        for pf in page_files:
            page_num = int(pf.stem.split("_")[1])
            regions = self.detect_page(str(pf), conf=conf)

            figure_regions = [
                r for r in regions
                if r["type"] in _EXTRACT_CLASSES and r["confidence"] >= conf
            ]
            figure_regions.sort(key=lambda r: r["bbox"][1])
            figure_regions = _deduplicate_boxes(figure_regions)

            if not figure_regions:
                continue

            img = Image.open(pf)
            for idx, region in enumerate(figure_regions, 1):
                x1, y1, x2, y2 = region["bbox"]
                pad = 5
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(img.width, x2 + pad)
                y2 = min(img.height, y2 + pad)

                crop = img.crop((x1, y1, x2, y2))
                fname = f"page{page_num}_fig{idx}.jpg"
                fpath = dest / fname
                crop.save(fpath, "JPEG", quality=95)

                all_crops.append({
                    "page": page_num,
                    "index": idx,
                    "type": region["type"],
                    "filename": fname,
                    "path": str(fpath),
                    "bbox": region["bbox"],
                    "size": crop.size,
                    "confidence": region["confidence"],
                })

            n_fig = sum(1 for r in figure_regions if r["type"] in _FIGURE_CLASSES)
            n_tab = sum(1 for r in figure_regions if r["type"] in _TABLE_CLASSES)
            parts = []
            if n_fig:
                parts.append(f"{n_fig} figure")
            if n_tab:
                parts.append(f"{n_tab} table")
            print(f"  第{page_num}页: {', '.join(parts)}")

        print(f"  共裁剪 {len(all_crops)} 张图表区域")
        return all_crops
