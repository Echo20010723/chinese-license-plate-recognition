"""Load models once and recognise the highest-confidence plate in each image."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETECTOR = ROOT / "detection/runs/ccpd2019_yolov8n_gpu_30ep/weights/best.pt"
DEFAULT_RECOGNIZER = ROOT / "recognition/checkpoints/seven_head_ccpd2019.pt"


class PlateRecognizer:
    """Seven-character Chinese plate inference without a CCPD dataset.

    Instances retain model weights for sequential calls. Use a separate instance
    per worker when integrating into an application with concurrent requests.
    """

    def __init__(self, detector=DEFAULT_DETECTOR, recognizer=DEFAULT_RECOGNIZER,
                 device="cpu", conf=0.25, imgsz=640):
        if not 0 < conf <= 1 or imgsz <= 0:
            raise ValueError("conf must be in (0, 1] and imgsz must be positive")
        for path in (detector, recognizer):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        from ultralytics import YOLO

        self.detector = YOLO(str(detector))
        self.recognizer_path = Path(recognizer)
        self.recognizer = None
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def predict(self, image_path):
        """Read an image and return a JSON-serialisable result; writes no files."""
        import cv2
        import numpy as np
        from recognition.inference import load_recognizer, recognise_plate

        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        encoded = np.fromfile(image_path, dtype=np.uint8)
        if not encoded.size:
            raise ValueError(f"Cannot decode empty image: {image_path}")
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode image: {image_path}")
        result = self.detector.predict(
            source=image, device=self.device, imgsz=self.imgsz,
            conf=self.conf, verbose=False, save=False,
        )[0]
        record = {"image": str(image_path), "detected": False, "bbox": None,
                  "detector_confidence": None, "recognition": None}
        if result.boxes is not None and len(result.boxes):
            index = int(result.boxes.conf.argmax().item())
            bbox = result.boxes.xyxy[index].detach().cpu().tolist()
            if self.recognizer is None:
                self.recognizer = load_recognizer(self.recognizer_path, self.device)
            record.update(detected=True, bbox=bbox,
                          detector_confidence=float(result.boxes.conf[index].item()),
                          recognition=recognise_plate(image, bbox, self.recognizer))
        return record
