"""Detect and recognise one seven-character plate; no training dataset required."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DETECTOR = ROOT / "detection/runs/ccpd2019_yolov8n_gpu_30ep/weights/best.pt"
RECOGNIZER = ROOT / "recognition/checkpoints/seven_head_spatial_residual_best.pt"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--detector", type=Path, default=DETECTOR)
    parser.add_argument("--recognizer", type=Path, default=RECOGNIZER)
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda:0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/prediction.json")
    args = parser.parse_args(argv)
    if not 0 < args.conf <= 1 or args.imgsz <= 0:
        parser.error("--conf must be in (0, 1] and --imgsz must be positive")
    for path in (args.image, args.detector, args.recognizer):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    if args.output.resolve() in {p.resolve() for p in (args.image, args.detector, args.recognizer)}:
        parser.error("Output must differ from the input image and checkpoints")

    import cv2
    import numpy as np
    from ultralytics import YOLO
    from recognition.inference import load_recognizer, recognise_plate

    # imdecode supports Windows paths containing non-ASCII characters.
    image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"Cannot decode image: {args.image}")
    result = YOLO(str(args.detector)).predict(
        source=image, device=args.device, imgsz=args.imgsz,
        conf=args.conf, verbose=False, save=False,
    )[0]
    record = {"image": str(args.image), "detected": False, "bbox": None,
              "detector_confidence": None, "recognition": None}
    if result.boxes is not None and len(result.boxes):
        index = int(result.boxes.conf.argmax().item())
        bbox = result.boxes.xyxy[index].detach().cpu().tolist()
        model = load_recognizer(args.recognizer, args.device)
        record.update(detected=True, bbox=bbox,
                      detector_confidence=float(result.boxes.conf[index].item()),
                      recognition=recognise_plate(image, bbox, model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, ensure_ascii=False, indent=2)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return record


if __name__ == "__main__":
    main()
