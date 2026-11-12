# Engineering validation

Checked on 2026-11-12 with Python 3.12.7 on Windows, CPU inference.

## Changes

- Added `predict.py` for dataset-independent, single-image detection and recognition.
  JSON output includes Unicode plate text, box coordinates and character confidences;
  no-detection results have explicit null fields.
- Extracted the checkpoint-compatible residual CNN into `recognition/model.py` and
  added preprocessing and inference in `recognition/inference.py`. Input remains
  RGB, 48 x 192, ImageNet normalisation, with inclusive bounding-box crop bounds.
- Configuration no longer fails on import when the external dataset is absent.
  Explicit dataset overrides are respected, and dataset operations validate them.
- Hardened CCPD annotation parsing against invalid coordinates, malformed vertices,
  invalid class indices and cross-platform path separators.
- Full/dev YOLO YAML files now select all test subsets rather than only the first one.
- Added automated tests, GitHub Actions, development instructions and attribution notes.
- Excluded raw datasets and generated outputs from Git tracking; retained final weights,
  training records, evaluation records, the original notebook and report.

## Checks performed

- `python -m pytest -q`: **16 passed**. Includes actual recognition checkpoint loading
  and synthetic-input inference, crop validation, annotation parsing, missing-dataset
  configuration, CLI help, highest-confidence box selection and no-detection JSON.
  Detection selection/no-detection tests use a controlled detector double.
- `python review_results.py`: passed. Checks 45,000 selected metadata rows, disjoint
  formal splits, 10,000 prediction records per evaluation and saved summary consistency.
- `python -m pip check`: no broken requirements in the verification environment.
- Real CPU CLI smoke run on `detection/results/prediction_gallery/base.jpg`:
  detector and recogniser loaded successfully and produced JSON. This image is an
  existing annotated gallery image; the smoke test is a functional check, not a new
  accuracy measurement. Output is ignored at `outputs/demo_prediction.json`.
- `python detection/predict_one.py --help`: works without the raw dataset.
- Detector-only CPU smoke inference saved an annotated image to
  `outputs/detector_smoke.jpg`, including support for the project's Unicode path.
- All Python files and 35 notebook code cells passed syntax parsing.

The local verification environment was created with `--system-site-packages` to reuse
installed PyTorch; NumPy and Ultralytics were installed inside `.venv`. Versions used:
PyTorch 2.3.1, torchvision 0.18.1a0, NumPy 1.26.4, OpenCV 4.11.0.86,
Ultralytics 8.4.211. This is a validation record, not a cross-platform lockfile.

No training, full raw-image evaluation or throughput benchmark was rerun. Existing
accuracy/FPS values remain the original group experiment results. GitHub Actions
results must be checked on GitHub after upload; local passing checks do not assert
that a remote run has occurred.
