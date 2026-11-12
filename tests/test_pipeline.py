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
@pytest.mark.parametrize("entrypoint", ["cli", "api"])
def test_cli_output_for_detection_and_no_detection(tmp_path, monkeypatch, detected, entrypoint):
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
    def load(*args):
        calls.append("load")
        return object()

    monkeypatch.setattr(recognition, "load_recognizer", load)
    monkeypatch.setattr(recognition, "recognise_plate", lambda *a, **k: {"text": "皖A01234"})
    if entrypoint == "cli":
        predict.main(["--image", str(image), "--detector", str(checkpoint),
                      "--recognizer", str(checkpoint), "--output", str(output)])
        result = json.loads(output.read_text(encoding="utf-8"))
    else:
        from pipeline import PlateRecognizer
        reader = PlateRecognizer(checkpoint, checkpoint)
        result = reader.predict(image)
        assert reader.predict(image) == result
        assert not output.exists()
    assert result["detected"] is detected
    if detected:
        assert result["bbox"] == [5, 6, 100, 40]
        assert result["recognition"]["text"] == "皖A01234"
        assert calls == ["load"]
    else:
        assert result["bbox"] is result["recognition"] is None
        assert calls == []


def test_empty_image_has_a_clear_error(tmp_path):
    from pipeline import PlateRecognizer
    empty = tmp_path / "empty.jpg"
    empty.touch()
    reader = PlateRecognizer.__new__(PlateRecognizer)
    with pytest.raises(ValueError, match="empty image"):
        reader.predict(empty)
