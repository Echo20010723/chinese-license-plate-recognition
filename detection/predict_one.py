"""Run the detector on one image and save only the highest-confidence box."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from common import RESULTS_DIR, optional_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    weights = Path(args.weights)
    image_path = Path(args.image)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    device = optional_device(args.device)
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read {image_path}")
    model = YOLO(str(weights))
    predict_args = {
        "source": image,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "verbose": False,
        "save": False,
    }
    if device is not None:
        predict_args["device"] = device

    result = model.predict(**predict_args)[0]

    selected_box = None
    confidence = 0.0
    if result.boxes is not None and len(result.boxes) > 0:
        confidences = result.boxes.conf.detach().cpu().numpy()
        best_index = int(np.argmax(confidences))
        confidence = float(confidences[best_index])
        selected_box = result.boxes.xyxy[best_index].detach().cpu().numpy().astype(int)

        x1, y1, x2, y2 = selected_box.tolist()
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            image,
            f"license_plate {confidence:.3f}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else RESULTS_DIR / f"prediction_{image_path.name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(output_path.suffix, image)
    if not success:
        raise OSError(f"OpenCV could not write annotated image to {output_path}")
    encoded.tofile(output_path)

    if selected_box is None:
        print("No box passed the confidence threshold.")
    else:
        print("Top-confidence box [x1, y1, x2, y2]:", selected_box.tolist())
        print(f"Confidence: {confidence:.4f}")
    print("Saved annotated image:", output_path)


if __name__ == "__main__":
    main()
