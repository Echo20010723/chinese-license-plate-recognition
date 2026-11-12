import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

import predict
import recognition.inference as recognition


@pytest.mark.parametrize("detected", [True, False])
def test_cli_output_for_detection_and_no_detection(tmp_path, monkeypatch, detected):
    image = tmp_path / "car.png"
    cv2.imencode(".png", np.zeros((48, 192, 3), dtype=np.uint8))[1].tofile(image)
    checkpoint = tmp_path / "test.pt"
    checkpoint.touch()
    output = tmp_path / "nested/result.json"

    class Boxes:
        conf = torch.tensor([0.4, 0.9])
        xyxy = torch.tensor([[1, 1, 2, 2], [5, 6, 100, 40]])

        def __len__(self):
            return 2 if detected else 0

    class Detector:
        def __init__(self, weights):
            pass

        def predict(self, **kwargs):
            return [SimpleNamespace(boxes=Boxes())]

    calls = []
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Detector))
    monkeypatch.setattr(recognition, "load_recognizer", lambda *a: calls.append("load"))
    monkeypatch.setattr(recognition, "recognise_plate", lambda *a, **k: {"text": "皖A01234"})
    predict.main(["--image", str(image), "--detector", str(checkpoint),
                  "--recognizer", str(checkpoint), "--output", str(output)])
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["detected"] is detected
    if detected:
        assert result["bbox"] == [5, 6, 100, 40]
        assert result["recognition"]["text"] == "皖A01234"
        assert calls == ["load"]
    else:
        assert result["bbox"] is result["recognition"] is None
        assert calls == []
