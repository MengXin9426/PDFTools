# PDFTools — 英文学术 PDF 自动翻译

将英文学术 PDF 自动翻译为中文，生成高质量 LaTeX 排版的 PDF 文档。

**开发者：沈阳理工大学装备工程学院深度学习实验室**

> 本项目专为学术研究设计，旨在提供高质量的英文学术论文自动翻译解决方案。

## 核心功能

- **智能文本提取**：基于 Qwen-VL-OCR 的结构化提取，生成 Markdown/LaTeX 格式
- **版面检测**：集成 DocLayout-YOLO 进行版面分析和图表区域检测
- **公式保护**：自动识别并保护数学公式，翻译时确保格式完整
- **大模型翻译**：支持阿里通义千问 Qwen 和本地 vLLM 后端
- **LaTeX 编译**：智能双栏/单栏自适应排版，自动生成高质量中文 PDF
- **质量检查**：基于 SSIM 和余弦相似度的翻译质量评估
- **批量处理**：支持页码范围选择和批量文档处理

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/snow-wind-001/PDFTools.git
cd PDFTools

# 安装依赖
pip install -e .

# 安装 LaTeX 编译环境（Ubuntu/Debian）
sudo apt install texlive-xetex fonts-noto-cjk
```

### 2. 配置 API

```bash
# 方式1：环境变量（推荐）
export QWEN_API_KEY="your-qwen-api-key"

# 方式2：编辑配置文件
cp config.yaml.example config.yaml
# 编辑 config.yaml 中的 api.qwen.api_key
```

### 3. 运行翻译

```bash
# 基础用法
python translate_pdf.py input/paper.pdf

# 指定页码范围
python translate_pdf.py input/paper.pdf --pages 1-5

# 使用本地 vLLM 后端
python translate_pdf.py input/paper.pdf --backend vllm

# 仅提取不翻译（测试用）
python translate_pdf.py input/paper.pdf --ocr-only
```

## 命令行参数

```bash
python translate_pdf.py PDF_FILE [选项]

必需参数:
  PDF_FILE                      输入的 PDF 文件路径

可选参数:
  -o, --output OUTPUT           输出目录（默认：output/<文件名>）
  --pages PAGES                 页码范围（如：1-3,5,8-10）
  --backend {qwen,vllm}        翻译后端选择
  --layout {auto,twocolumn,onecolumn}  排版模式
  --dpi DPI                    PDF 渲染 DPI（默认：300）
  --ocr-model MODEL            OCR 模型名称
  --translate-model MODEL      翻译模型名称
  
流程控制:
  --ocr-only                   仅执行 OCR 提取
  --from-ocr                   跳过 OCR，从已有结果开始
  --no-latex                   跳过 LaTeX 编译
  --no-quality                 跳过质量检查
  --no-layout-detect           跳过版面检测
```

## 配置说明

### 环境变量

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `QWEN_API_KEY` | 阿里通义千问 API 密钥 | `sk-xxxxx` |
| `PDFTOOLS_MAX_PAGES` | 限制处理页数（测试用） | `5` |

### 配置文件

编辑 `config.yaml` 可配置：
- **翻译后端**：`qwen`（云端）或 `vllm`（本地）
- **API 设置**：密钥、模型名称、超时等
- **版面检测**：模型路径、置信度阈值
- **LaTeX 排版**：编译器、排版模式、字体等
- **质量检查**：SSIM 阈值、输出 DPI

## 输出结构

```
output/
├── <PDF名>/                    # 主输出目录
│   ├── english.md              # OCR 提取的英文原文
│   ├── chinese.md              # 翻译后的中文 Markdown
│   ├── translated_zh.txt       # 翻译后的纯文本
│   ├── document.tex            # LaTeX 源码
│   ├── ocr/                    # OCR 中间结果
│   │   ├── page_images/        # 页面图片
│   │   └── ...                 # OCR 详细输出
│   └── logs/                   # 处理日志
└── result/<PDF名>/             # LaTeX 编译结果
    ├── document.pdf            # 最终翻译 PDF
    ├── document.tex            # 编译用 LaTeX 源码
    └── images/                 # 提取的图表
```

## 项目架构

```
PDFTools/
├── 主程序
│   ├── translate_pdf.py          # 完整翻译流程（推荐）
│   ├── pipeline_test.py          # 简化版流程
│   └── config.yaml               # 配置文件
├── 核心模块 (skills/)
│   ├── qwen_vl_ocr.py           # Qwen-VL-OCR 文本提取
│   ├── ocr_to_chunks.py         # OCR 结果解析
│   ├── layout_detector/         # DocLayout-YOLO 版面检测
│   ├── translator/              # 翻译引擎
│   │   ├── qwen_translator.py   # Qwen 翻译器
│   │   └── glossary.py          # 术语词典
│   ├── latex_builder/           # LaTeX 文档构建
│   ├── quality_checker/         # 质量评估
│   └── advanced_extractor/      # 高级提取功能
├── 数据目录
│   ├── input/                   # 输入 PDF 文件
│   ├── output/                  # 输出结果
│   └── models/                  # 模型权重缓存
└── 项目配置
    ├── pyproject.toml           # Python 包配置
    ├── .gitignore              # Git 忽略规则
    └── README.md               # 项目文档
```

## 系统要求

### 运行环境
- **Python**：>= 3.10
- **操作系统**：Linux（推荐 Ubuntu 20.04+）、macOS、Windows
- **内存**：建议 8GB+（处理大文档时）
- **存储**：至少 5GB 可用空间（包含模型缓存）

### 依赖软件

```bash
# LaTeX 编译环境（Ubuntu/Debian）
sudo apt update
sudo apt install texlive-xetex fonts-noto-cjk texlive-fonts-extra

# macOS（使用 Homebrew）
brew install --cask mactex
brew install font-noto-sans-cjk

# 验证安装
xelatex --version
```

### GPU 支持（可选）
- **CUDA**：>= 11.8（用于本地 vLLM 推理）
- **显存**：建议 12GB+（7B 模型）/ 24GB+（13B 模型）

## 技术特色

- **🚀 高效处理**：Qwen-VL-OCR 单次调用完成文本和公式提取
- **🎯 精准识别**：DocLayout-YOLO 实现像素级版面检测
- **🛡️ 公式保护**：智能识别数学表达式，翻译时完整保留
- **🔄 双模式支持**：云端 Qwen API 和本地 vLLM 部署可选
- **📐 智能排版**：自适应双栏/单栏，处理复杂学术文档格式
- **🔍 质量保证**：多维度质量评估，确保翻译准确性

## 使用案例

适用于以下学术场景：
- 📄 **期刊论文翻译**：IEEE、ACM、Springer 等英文论文
- 📚 **教材资料翻译**：技术教材、参考手册
- 📊 **报告文档翻译**：研究报告、白皮书
- 🧮 **数学文档翻译**：包含复杂公式的学术资料

## 开发团队

**沈阳理工大学装备工程学院深度学习实验室**

本项目由深度学习实验室开发维护，专注于 AI 在学术文档处理领域的应用研究。

## 许可证

MIT License - 详见 LICENSE 文件

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 联系我们

- 🏫 **单位**：沈阳理工大学装备工程学院深度学习实验室
- 📧 **邮箱**：[联系邮箱]
- 🐛 **问题反馈**：[GitHub Issues](https://github.com/snow-wind-001/PDFTools/issues)
