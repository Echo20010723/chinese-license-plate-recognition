"""Reusable recognition with the original crop and normalisation protocol."""
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from ccpd_annotations import ADS, ALPHABETS, PROVINCES
from recognition.model import SevenHeadRecognizer

POSITION_TABLES = [PROVINCES, ALPHABETS] + [ADS] * 5
NUM_CLASSES = [len(table) for table in POSITION_TABLES]
INPUT_SIZE = (48, 192)
EVALUATION_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def crop_bounding_box(image, bbox):
    """Use inclusive pixel bounds, matching the recorded notebook evaluation."""
    if image is None or image.size == 0:
        raise ValueError("Image is empty")
    values = np.asarray(bbox, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Bounding box must contain four finite coordinates")
    x1, y1, x2, y2 = map(int, values)
    height, width = image.shape[:2]
    if x2 < x1 or y2 < y1 or x2 < 0 or y2 < 0 or x1 >= width or y1 >= height:
        raise ValueError("Bounding box is reversed or outside the image")
    x1, x2 = max(0, x1), min(width - 1, x2)
    y1, y2 = max(0, y1), min(height - 1, y2)
    return image[y1:y2 + 1, x1:x2 + 1]


def load_recognizer(weights: str | Path, device="cpu"):
    """Load the bundled tensor/primitive checkpoint without arbitrary unpickling."""
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    if list(checkpoint["num_classes"]) != NUM_CLASSES:
        raise ValueError("Checkpoint vocabulary does not match the seven-head model")
    if tuple(checkpoint["input_size"]) != INPUT_SIZE:
        raise ValueError("Checkpoint input size does not match preprocessing")
    model = SevenHeadRecognizer(NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


@torch.inference_mode()
def recognise_plate(image_bgr, bbox, model):
    crop = crop_bounding_box(image_bgr, bbox)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    device = next(model.parameters()).device
    outputs = model(EVALUATION_TRANSFORM(rgb).unsqueeze(0).to(device))
    indices = [int(output.argmax(dim=1).item()) for output in outputs]
    confidences = [float(output.softmax(dim=1).max().item()) for output in outputs]
    return {
        "text": "".join(table[index] for table, index in zip(POSITION_TABLES, indices)),
        "indices": indices,
        "character_confidences": confidences,
        "mean_confidence": sum(confidences) / len(confidences),
    }
