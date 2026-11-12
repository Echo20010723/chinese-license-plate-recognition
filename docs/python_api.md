# Python API

Use Python from the repository root; this repository is not currently published as
a pip package. `PlateRecognizer` retains its models across sequential predictions.

```python
import json
from pathlib import Path
from pipeline import PlateRecognizer

reader = PlateRecognizer(device="cpu", conf=0.25, imgsz=640)
for image_path in sorted(Path("my_images").glob("*.jpg")):
    result = reader.predict(image_path)
    print(json.dumps(result, ensure_ascii=False))
```

`predict()` accepts a string or `pathlib.Path` to an image. It returns a plain,
JSON-serialisable dictionary and does not write files. Recognition weights are loaded
on the first detected plate and reused. Use a separate instance per concurrent worker.

## Constructor

```python
reader = PlateRecognizer(
    detector="detection/runs/ccpd2019_yolov8n_gpu_30ep/weights/best.pt",
    recognizer="recognition/checkpoints/seven_head_ccpd2019.pt",
    device="cpu",
    conf=0.25,
    imgsz=640,
)
```

Omit checkpoint arguments to use paths resolved relative to the installed repository.
Explicit paths are relative to the current working directory. Recognition checkpoints
must have the architecture, vocabulary and `(48, 192)` input size used by the included model.

## Return fields

| Field | Type | Meaning |
| --- | --- | --- |
| `image` | string | Input image path |
| `detected` | boolean | A plate passed the confidence threshold |
| `bbox` | four numbers or null | Highest-confidence box in original-image pixels |
| `detector_confidence` | number or null | Selected detector confidence |
| `recognition` | object or null | Seven-character recognition result |
| `recognition.text` | string | Chinese plate text |
| `recognition.indices` | seven integers | Character indices in the CCPD position vocabularies |
| `recognition.character_confidences` | seven numbers | Per-head maximum softmax probabilities |
| `recognition.mean_confidence` | number | Arithmetic mean of the seven probabilities |

With no detected plate, box, detector confidence and recognition are all `None`.
Confidence values are not calibrated probabilities of a correct whole plate.

## Errors and boundaries

Missing inputs/checkpoints raise `FileNotFoundError`; invalid thresholds, invalid images
or incompatible recognition metadata raise `ValueError`. PyTorch/Ultralytics may raise
their own device or architecture errors. Handle these at your application's boundary.

Each call selects one plate. The included recogniser predicts exactly seven characters;
it does not infer whether a detected plate is an eight-character new-energy plate.
No raw CCPD dataset is required for this API.
