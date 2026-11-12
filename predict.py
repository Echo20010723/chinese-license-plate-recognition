"""Detect and recognise one seven-character plate; no training dataset required."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from pipeline.inference import DEFAULT_DETECTOR, DEFAULT_RECOGNIZER, PlateRecognizer

ROOT = Path(__file__).resolve().parent
DETECTOR = DEFAULT_DETECTOR
RECOGNIZER = DEFAULT_RECOGNIZER


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

    try:
        pipeline = PlateRecognizer(args.detector, args.recognizer, args.device, args.conf, args.imgsz)
        record = pipeline.predict(args.image)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, ensure_ascii=False, indent=2)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return record


if __name__ == "__main__":
    main()
