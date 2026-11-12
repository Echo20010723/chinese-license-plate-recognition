# 中国车牌检测与识别

[English](README.md) · [详细安装指南](docs/getting_started.md) · [Python 接口](docs/python_api.md) · [复现实验](docs/reproduce_experiments.md)

使用 **YOLOv8n 检测器 + 七头残差 CNN 识别器**，完成车辆图片的车牌定位与七位字符识别。仓库自带训练好的权重，处理自己的图片不需要下载 CCPD 数据集。

原课程小组成员：**Haoqing Gu、Junwen Xue、Zezheng Yu、Ye Li、Xiaoyang Chen**。报告采用会议论文格式，不代表已在 IEEE 发表。

## 快速运行

使用 **Python 3.10–3.12**：

```bash
git clone https://github.com/Echo20010723/chinese-license-plate-recognition.git
cd chinese-license-plate-recognition
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux 激活环境：

```bash
source .venv/bin/activate
```

安装依赖并运行仓库示例：

```bash
python -m pip install -r requirements.txt
python predict.py --image detection/results/prediction_gallery/base.jpg --output outputs/example.json
```

示例是已有标注的实验图片，用于确认流程可运行。换成自己的图片：

```bash
python predict.py --image path/to/car.jpg --device cpu --output outputs/my_plate.json
```

结果保存为 UTF-8 JSON，包括检测框、检测置信度、中文车牌文本与逐字符置信度。未检测到车牌时 `detected` 为 `false`，检测框和识别结果为 `null`。GPU 使用 `--device cuda:0`，先安装与硬件匹配的 PyTorch。

## 在自己的代码里使用

在仓库根目录运行，复用同一个实例可避免反复加载模型：

```python
from pipeline import PlateRecognizer

reader = PlateRecognizer(device="cpu")
result = reader.predict("path/to/car.jpg")
if result["detected"]:
    print(result["recognition"]["text"])
```

详细参数与字段见 [Python 接口说明](docs/python_api.md)。输出示例、中文路径支持、安装故障处理见 [入门指南](docs/getting_started.md)。

## 从哪里开始阅读

| 想做什么 | 对应入口 |
| --- | --- |
| 运行自己的车辆图片 | `predict.py` |
| 集成到程序中 | `pipeline/inference.py` |
| 理解识别网络与预处理 | `recognition/model.py`、`recognition/inference.py` |
| 解析 CCPD 标注 | `ccpd_annotations.py` |
| 学习完整实验流程 | `notebooks/ccpd_workflow.ipynb` |
| 查看图文成果 | `docs/project_overview.html`，下载后用浏览器打开 |
| 阅读原始报告 | `docs/project_report.pdf` |
| 复现训练与评测 | [复现指南](docs/reproduce_experiments.md) |
| 校验已有实验记录 | `python validate_results.py`，仅需 Python 标准库 |

原始 CCPD 图片不包含在仓库中；`data_preparation/` 保留固定划分列表和元数据。检测、识别、端到端实验结果分别位于 `detection/results/`、`recognition/results/`、`pipeline/results/`。原始实验记录的长文件名保留了实验来源，便于追溯。

## 实验结果与边界

固定种子 9444，训练 / 验证 / 测试分别为 30,000 / 5,000 / 10,000 张图片。记录的 Base 检测 mAP@0.5 为 **99.5%**，整体端到端整牌准确率为 **70.41%**，同时要求定位正确的严格准确率为 **66.35%**。记录的 **47.22 FPS** 来自 RTX 3060 Laptop GPU，不能视为任意电脑或单张 CLI 的性能。

以上均为原小组实验记录。工程整理没有重新训练或刷新指标；不同评测协议的区别见 [评测说明](docs/evaluation.md)。当前程序每张图片选择置信度最高的一个车牌，识别器仅支持七位中国车牌，不支持八位新能源车牌。模糊、光照和透视变化会降低准确率。

## 测试与贡献

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python validate_results.py
```

GitHub Actions 自动检查 Python 3.10 和 3.12。[贡献说明](CONTRIBUTING.md)、[验证记录](docs/validation.md)、[文件重命名对照表](docs/file_naming.md)可帮助理解项目结构。

保留原始小组署名；公开仓库未额外授予全项目开源许可证，上游依赖、权重和数据集仍适用各自条款，详见 [来源与授权说明](THIRD_PARTY_NOTICES.md)。
