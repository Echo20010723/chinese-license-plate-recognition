# Car License Plates Detection and Recognition

A course group project for Chinese license plate detection and recognition on CCPD2019.
YOLOv8n localises the plate; a seven-head residual CNN predicts its seven characters.

**Team:** Haoqing Gu, Junwen Xue, Zezheng Yu, Ye Li, Xiaoyang Chen.

## 单张图片检测与识别 / Single-image inference

支持 **Python 3.10–3.12**。使用项目自带权重，无需下载 CCPD 数据集即可识别自己的车辆图片。
当前识别器仅支持七位中国车牌；八位新能源车牌可用于检测，但此识别器不支持。

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python predict.py --image path/to/car.jpg --device cpu --output outputs/prediction.json
```

输出 JSON 包含检测框、检测置信度、车牌中文文本及逐字符置信度。
未检测到车牌时返回 `detected: false`，检测框和识别结果为 `null`。
GPU 推理使用 `--device cuda:0`，并先安装与硬件匹配的 PyTorch。
命令选择置信度最高的一个车牌，适合单车牌图片；置信度不是经过校准的正确概率。

```json
{
  "image": "path/to/car.jpg",
  "detected": true,
  "bbox": [100.0, 200.0, 300.0, 250.0],
  "detector_confidence": 0.95,
  "recognition": {
    "text": "皖A01234",
    "indices": [0, 0, 24, 25, 26, 27, 28],
    "character_confidences": [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93],
    "mean_confidence": 0.96
  }
}
```

以上 JSON 仅用于说明格式，并非新增实验结果。命令从任意工作目录运行时，
请使用 `predict.py` 的绝对路径；默认权重始终相对于项目目录定位。

![Detection examples](detection/results/prediction_gallery/base.jpg)

## 快速查看 / Quick review

求职作品展示：先打开 `PROJECT_OVERVIEW.html` 查看项目概览、指标和效果图，再阅读
`report/Project_Report.pdf`。代码入口为同名 Notebook。查看已有结果不需要安装深度学习环境。
本项目为小组课程作品，报告采用会议论文格式，不代表已经在 IEEE 会议发表。

1. Open `PROJECT_OVERVIEW.html` in a browser for the saved results and example images.
2. Read `report/Project_Report.pdf` for methods, experiments, discussion and references.
3. Open `Car_License_Plates_Detection_and_Recognition.ipynb` for the complete workflow.
4. Run `python review_results.py` to print and validate saved results using the standard library.

## Directory guide

| Path | Contents |
| --- | --- |
| `Car_License_Plates_Detection_and_Recognition.ipynb` | Data preparation, detection, recognition and combined evaluation |
| `config.py`, `parse_tool.py` | Dataset configuration and CCPD filename annotation parser |
| `data_preparation/` | Fixed split lists and metadata for 45,000 selected images |
| `detection/` | Detector scripts, pretrained/final weights, training logs, metrics and galleries |
| `predict.py` | Dataset-independent detection + recognition CLI with JSON output |
| `recognition/` | Reusable model/inference modules, final checkpoint, training history and evaluation summaries; training remains in the notebook |
| `pipeline/results/` | Combined predictions, summary metrics and failure analysis |
| `report/` | Original project report |
| `docs/` | Evaluation definitions and packaging notes |

## Recorded results and scope

- Data: 30,000 training, 5,000 validation and 10,000 test images, fixed seed 9444.
- Test sets: Base (3,000); Blur, Challenge, DB, FN, Rotate, Tilt and Weather (1,000 each).
- Detector-only Base: mAP@0.5 **99.5%**; localisation accuracy at IoU > 0.7 **99.2%**.
- Detector-only localisation accuracy across all test images: **90.52%** (confidence 0.001).
- Combined pipeline: exact-plate accuracy **70.41%**; strict end-to-end accuracy **66.35%**.
- Combined pipeline speed: **47.22 FPS**, recorded on an NVIDIA RTX 3060 Laptop GPU.
- Separate detector-only speed benchmark: **42.51 FPS**, batch 1, 500 images after 20 warmups.

These are different evaluation protocols, not interchangeable accuracy or speed claims.
See `docs/EVALUATION.md`. The experiment uses a selected subset, not the complete CCPD benchmark.

## Environment setup

Use an isolated Python environment with a compatible PyTorch/torchvision installation.
Install the appropriate PyTorch build for your hardware first, then install the remaining
dependencies in the same environment:

```bash
python -m venv .venv
# Activate .venv with the command appropriate for your operating system.
python -m pip install -r requirements.txt
python -m ipykernel install --user --name plate-project --display-name "Plate Project"
```

Open the notebook in VS Code or Jupyter using that environment and run cells in order.
Dependencies are version ranges from the original project, not a reproducibility lockfile.
Notebook outputs were cleared for a smaller, portable submission; saved results remain in CSVs,
figures and the report. Running all cells performs substantial data checks and evaluation.

## Dataset required for reruns

The raw CCPD2019 image dataset is external and is not bundled. Obtain it from the original
dataset project: https://github.com/detectRecog/CCPD and follow its distribution terms.
Place the `CCPD2019` directory beside `config.py`, with `splits/` and the original
`ccpd_base/`, `ccpd_blur/`, etc. subdirectories inside it. Alternatively set `CCPD_DATA_ROOT`
to that directory. Set the variable before starting the notebook kernel.

The bundled split lists and metadata preserve the exact selected images. The notebook's
first part can regenerate them from the same dataset. It also checks duplicate paths,
overlaps, annotation bounds and label coverage. The 3,000-image development set is drawn
from training and is not an independent test set.

## Detection commands

Run from the project root after installing dependencies and providing the dataset.
The bundled metadata is sufficient to prepare the YOLO layout:

```bash
python detection/prepare_yolo_dataset.py --mode auto
python detection/evaluate_detector.py --weights detection/runs/ccpd2019_yolov8n_gpu_30ep/weights/best.pt --subsets all --with-map --device cpu
```

Use `--device 0` for a compatible CUDA GPU. To retrain the detector with the recorded setup:

```bash
python detection/train_detector.py --data detection/yolo_dataset/ccpd_full.yaml --model detection/yolov8n.pt --epochs 30 --imgsz 640 --batch 16 --device 0 --name ccpd2019_yolov8n_gpu_30ep
```

The final detector and final recogniser weights are included, so the notebook reuses them
when present. Recognition training, recognition-only testing and combined evaluation are
implemented in notebook Parts 3 and 4. Training/evaluation can overwrite their generated
results; keep a copy of the submitted archive if you want to retain the recorded measurements.

## Attribution and limitations

This submission preserves team authorship. Libraries and research references are acknowledged
in the report and original code dependencies. It assumes one seven-character plate per image.
Recognition under blur, challenging lighting and perspective remains the main limitation.
The original metrics have not been replaced by new training or benchmark measurements.
See `docs/PACKAGING_NOTES.md` for the original clean-up scope and
`docs/VALIDATION.md` for subsequent engineering checks.

## Development and validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python review_results.py
```

GitHub Actions runs tests on Python 3.10 and 3.12, including loading the bundled
recognition checkpoint, inference on synthetic input, annotation validation and saved-result checks.
It does not download the dataset or retrain models. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Data and attribution

The repository excludes raw datasets, virtual environments and generated inference outputs.
The local `Foggy-Hazy License Plates Images` directory is not part of the recorded
CCPD2019 experiment and is not uploaded. Training checkpoints, selected split lists,
evaluation records and the original report are retained.

Original team authorship remains unchanged. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for dataset/library sources and licensing scope. No new project-wide open-source license is asserted.
