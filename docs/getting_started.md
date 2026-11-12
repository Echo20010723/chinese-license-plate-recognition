# Getting started / 安装与运行

## 1. Install

Clone the repository and create an isolated Python 3.10–3.12 environment as shown
in the [English README](../README.md) or [中文说明](../README.zh-CN.md).
Run all example commands from the repository root.

```bash
python -m pip install -r requirements.txt
python predict.py --help
```

The requirements are compatible version ranges, not a frozen training environment.
For GPU use, install the appropriate PyTorch/torchvision pair for your CUDA hardware
before installing the requirements. Default inference uses CPU.

If PowerShell activation is restricted, you can use the environment directly without
changing execution policy:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe predict.py --image detection/results/prediction_gallery/base.jpg
```

## 2. Run a bundled example

```bash
python predict.py --image detection/results/prediction_gallery/base.jpg --output outputs/example.json
```

This is an existing annotated gallery image. It verifies the pipeline runs; it is
not a clean benchmark input and its predicted text should not be treated as an accuracy claim.
For your own images, use an ordinary JPEG or PNG containing a visible seven-character
Chinese plate. Paths containing Chinese characters and spaces are supported; quote paths
with spaces in the shell.

```bash
python predict.py --image "my images/car.jpg" --output outputs/car.json
```

## 3. Understand the result

Illustrative schema, not a measured prediction:

```json
{
  "image": "my images/car.jpg",
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

Box coordinates are `[x1, y1, x2, y2]` in original-image pixels. All confidence
values are in `[0, 1]`; they are not calibrated correctness probabilities.
No detection is a successful result with `detected: false`, `bbox: null`,
`detector_confidence: null` and `recognition: null`.

The output file is overwritten when you reuse the same output path.
The CLI rejects output paths equal to the input image or model checkpoints.

## Options

| Flag | Default | Purpose |
| --- | --- | --- |
| `--image` | Required | Image file to read |
| `--output` | `outputs/prediction.json` in the repository | JSON destination |
| `--device` | `cpu` | PyTorch device, e.g. `cuda:0` |
| `--conf` | `0.25` | Detector threshold in `(0, 1]` |
| `--imgsz` | `640` | Positive detector input size |
| `--detector` | Included trained YOLO checkpoint | Custom compatible detector |
| `--recognizer` | Included seven-head CNN checkpoint | Custom compatible recogniser |

Default weights resolve relative to the repository, including when `predict.py`
is invoked by absolute path from another directory. Explicit image/output paths
are resolved relative to your terminal's current directory.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `ModuleNotFoundError` | Install requirements using the same Python executable that runs the command |
| NumPy / PyTorch binary compatibility errors | Use the project's `numpy<2` constraint and an isolated environment |
| CUDA is unavailable | Run with `--device cpu`, or install a hardware-compatible PyTorch build |
| File not found | Check the current directory and quote paths containing spaces |
| Cannot decode image | Use a valid, nonempty JPEG/PNG rather than a renamed document |
| `detected: false` | Check image visibility; optionally lower `--conf`, accepting more false detections |
| Wrong text | The recogniser supports seven characters and is sensitive to blur, perspective and lighting |
| Notebook asks for dataset | Inference is dataset-independent; training/full evaluation needs external CCPD images |

For repeated calls, use the [Python API](python_api.md). For experiments, see
[reproduction](reproduce_experiments.md).
