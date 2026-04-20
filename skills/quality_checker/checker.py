"""
质量检查器
比较原始PDF和生成PDF的格式一致性
"""

import os
import json
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image


class QualityChecker:
    """质量检查器"""

    def __init__(self, output_dir: str = "output"):
        """
        初始化检查器

        Args:
            output_dir: 输出文件夹
        """
        self.output_dir = Path(output_dir)

    def check_quality(self, original_pdf: str, generated_pdf: str, pdf_name: str) -> Dict:
        """
        检查PDF质量

        Args:
            original_pdf: 原始PDF路径
            generated_pdf: 生成的PDF路径
            pdf_name: PDF文件名

        Returns:
            质量检查报告
        """
        print(f"\n质量检查: {pdf_name}")

        gen_path = Path(generated_pdf)
        if not gen_path.exists() or gen_path.suffix.lower() != ".pdf":
            report = {
                "pdf_name": pdf_name,
                "total_pages": 0,
                "skipped": True,
                "reason": "未找到编译后的生成 PDF（可能未安装 xelatex 或编译失败）",
                "average_ssim": 0.0,
                "quality_level": "Unknown",
            }
            report_path = self.output_dir / "result" / pdf_name / "quality_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"  ⊘ 跳过页面比对: {report['reason']}")
            print(f"\n✓ 质量报告已保存: {report_path}")
            return report

        # 创建对比图片目录
        compare_dir = self.output_dir / "result" / pdf_name / "comparison"
        compare_dir.mkdir(parents=True, exist_ok=True)

        # 转换PDF页面为图片
        print("  → 转换PDF页面为图片...")
        original_pages = self._pdf_to_pages(original_pdf, compare_dir / "original")
        generated_pages = self._pdf_to_pages(str(gen_path), compare_dir / "generated")

        # 比较页面
        print("  → 比较页面相似度...")
        comparison_results = []

        min_pages = min(len(original_pages), len(generated_pages))

        for i in range(min_pages):
            result = self._compare_pages(
                original_pages[i],
                generated_pages[i],
                compare_dir / f"page_{i + 1}_comparison.png"
            )
            comparison_results.append(result)
            print(f"  ✓ 页面 {i + 1}: SSIM={result['ssim']:.3f}, Cosine={result['cosine']:.3f}")

        # 生成报告
        report = self._generate_report(comparison_results, pdf_name)

        # 保存报告
        report_path = self.output_dir / "result" / pdf_name / "quality_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n✓ 质量报告已保存: {report_path}")

        return report

    def _pdf_to_pages(self, pdf_path: str, output_dir: Path) -> List[str]:
        """
        将PDF转换为图片

        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录

        Returns:
            图片文件路径列表
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import pymupdf

            doc = pymupdf.open(pdf_path)
            page_images = []

            for page_num, page in enumerate(doc):
                # 渲染为图片
                pix = page.get_pixmap(dpi=150)
                image_path = output_dir / f"page_{page_num + 1}.png"
                pix.save(str(image_path))
                page_images.append(str(image_path))

            doc.close()
            return page_images

        except Exception as e:
            print(f"  ⚠ PDF转换失败: {e}")
            return []

    def _compare_pages(self, original_path: str, generated_path: str, output_path: Path) -> Dict:
        """
        比较两个页面图片

        Args:
            original_path: 原始页面路径
            generated_path: 生成页面路径
            output_path: 对比图输出路径

        Returns:
            比较结果字典
        """
        # 读取图片
        original = cv2.imread(original_path)
        generated = cv2.imread(generated_path)

        if original is None or generated is None:
            return {
                'ssim': 0.0,
                'cosine': 0.0,
                'mse': float('inf'),
                'error': 'Cannot read images'
            }

        # 调整大小（使尺寸一致）
        height, width = original.shape[:2]
        generated = cv2.resize(generated, (width, height))

        # 转换为灰度图
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        generated_gray = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)

        # 计算SSIM
        ssim_value, _ = ssim(original_gray, generated_gray, full=True)

        # 计算余弦相似度（基于图像特征）
        original_features = original_gray.flatten().reshape(1, -1)
        generated_features = generated_gray.flatten().reshape(1, -1)
        cosine_value = cosine_similarity(original_features, generated_features)[0][0]

        # 计算MSE
        mse = np.mean((original_gray.astype(float) - generated_gray.astype(float)) ** 2)

        # 生成对比图
        self._create_comparison_image(original, generated, str(output_path))

        return {
            'ssim': float(ssim_value),
            'cosine': float(cosine_value),
            'mse': float(mse)
        }

    def _create_comparison_image(self, original: np.ndarray, generated: np.ndarray, output_path: str):
        """
        创建对比图片

        Args:
            original: 原始图片
            generated: 生成图片
            output_path: 输出路径
        """
        # 并排显示
        height, width = original.shape[:2]
        comparison = np.zeros((height, width * 2, 3), dtype=np.uint8)
        comparison[:, :width] = original
        comparison[:, width:] = generated

        # 添加标签
        cv2.putText(comparison, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Generated", (width + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imwrite(output_path, comparison)

    def _generate_report(self, results: List[Dict], pdf_name: str) -> Dict:
        """
        生成质量报告

        Args:
            results: 比较结果列表
            pdf_name: PDF文件名

        Returns:
            报告字典
        """
        if not results:
            return {
                'pdf_name': pdf_name,
                'total_pages': 0,
                'error': 'No comparison results'
            }

        # 计算平均指标
        avg_ssim = sum(r['ssim'] for r in results) / len(results)
        avg_cosine = sum(r['cosine'] for r in results) / len(results)
        avg_mse = sum(r['mse'] for r in results) / len(results)

        # 评估质量等级
        quality_level = self._evaluate_quality(avg_ssim, avg_cosine)

        return {
            'pdf_name': pdf_name,
            'total_pages': len(results),
            'average_ssim': avg_ssim,
            'average_cosine': avg_cosine,
            'average_mse': avg_mse,
            'quality_level': quality_level,
            'page_details': results
        }

    def _evaluate_quality(self, ssim: float, cosine: float) -> str:
        """
        评估质量等级

        Args:
            ssim: SSIM值
            cosine: 余弦相似度

        Returns:
            质量等级
        """
        if ssim > 0.9 and cosine > 0.95:
            return "优秀 (Excellent)"
        elif ssim > 0.8 and cosine > 0.9:
            return "良好 (Good)"
        elif ssim > 0.7 and cosine > 0.85:
            return "一般 (Fair)"
        else:
            return "较差 (Poor)"
