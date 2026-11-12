"""Review and cross-check saved project results without ML packages or raw images."""
from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_csv(relative_path):
    with (ROOT / relative_path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_list(name):
    return (ROOT / "data_preparation" / name).read_text(encoding="utf-8-sig").splitlines()


def close(actual, expected, name):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError(f"Inconsistent {name}: {actual} vs {expected}")


def truth(value):
    return value.strip().lower() == "true"


def main():
    subsets = ["base", "blur", "challenge", "db", "fn", "rotate", "tilt", "weather"]
    split_files = {"train": "train_30k.txt", "validation": "val_5k.txt"}
    split_files.update({s: f"test_{s}_{'3k' if s == 'base' else '1k'}.txt" for s in subsets})
    splits = {name: read_list(filename) for name, filename in split_files.items()}
    expected_sizes = {"train": 30000, "validation": 5000, "base": 3000}
    expected_sizes.update({s: 1000 for s in subsets if s != "base"})
    seen = set()
    for name, paths in splits.items():
        if len(paths) != expected_sizes[name] or len(set(paths)) != len(paths):
            raise ValueError(f"Invalid size or duplicate paths in {name}")
        if seen.intersection(paths):
            raise ValueError(f"Overlapping formal split: {name}")
        seen.update(paths)
    if not set(read_list("train_dev_3k.txt")).issubset(splits["train"]):
        raise ValueError("Development list is not a training subset")
    metadata = read_csv("data_preparation/ccpd_selected_metadata.csv")
    if len(metadata) != 45000 or {r["relative_path"] for r in metadata} != seen:
        raise ValueError("Metadata and selected split paths disagree")

    detection = read_csv("detection/results/ccpd2019_gpu_30ep_original_seed_summary.csv")
    detection_predictions = read_csv("detection/results/ccpd2019_gpu_30ep_original_seed_per_image.csv")
    pipeline = read_csv("pipeline/results/end_to_end_summary.csv")
    predictions = read_csv("pipeline/results/end_to_end_predictions.csv")
    if len(detection_predictions) != 10000 or len(predictions) != 10000:
        raise ValueError("Expected exactly 10,000 predictions in each evaluation")
    for row in detection:
        group = [p for p in detection_predictions if p["subset"] == row["subset"]]
        if len(group) != int(row["images"]):
            raise ValueError("Detector prediction count mismatch")
        correct = sum(truth(p["correct_iou_gt_0_7"]) for p in group)
        close(correct, row["ccpd_correct_iou_gt_0_7"], "detector correct count")
        close(correct / len(group), row["ccpd_accuracy_iou_gt_0_7"], "detector accuracy")
    for row in pipeline:
        group = predictions if row["subset"] == "overall" else [p for p in predictions if p["subset"] == row["subset"]]
        if len(group) != int(row["images"]):
            raise ValueError("Pipeline prediction count mismatch")
        close(sum(truth(p["detected"]) for p in group), row["detected"], "pipeline detections")
        exact = sum(truth(p["exact_plate_correct"]) for p in group)
        strict = sum(truth(p["exact_plate_correct"]) and truth(p["localisation_correct"]) for p in group)
        loc = sum(truth(p["localisation_correct"]) for p in group)
        close(exact / len(group), row["end_to_end_exact_plate_accuracy"], "exact accuracy")
        close(strict / len(group), row["strict_end_to_end_accuracy"], "strict accuracy")
        close(loc / len(group), row["localisation_accuracy_iou_gt_0_7"], "localisation accuracy")
        close(len(group) / float(row["elapsed_seconds"]), row["fps"], "saved throughput")
    for checkpoint in ["detection/yolov8n.pt", "detection/runs/ccpd2019_yolov8n_gpu_30ep/weights/best.pt",
                       "recognition/checkpoints/seven_head_ccpd2019.pt"]:
        if not (ROOT / checkpoint).is_file():
            raise FileNotFoundError(checkpoint)

    overall = next(r for r in pipeline if r["subset"] == "overall")
    detector_acc = sum(int(r["ccpd_correct_iou_gt_0_7"]) for r in detection) / 10000
    print("Car License Plates Detection and Recognition")
    print("Saved-data checks passed: 45,000 metadata rows; disjoint formal splits; 10,000 predictions per evaluation.")
    print(f"Detector-only localisation accuracy (IoU > 0.7, conf 0.001): {detector_acc:.2%}")
    print(f"Pipeline exact-plate accuracy: {float(overall['end_to_end_exact_plate_accuracy']):.2%}")
    print(f"Pipeline strict end-to-end accuracy: {float(overall['strict_end_to_end_accuracy']):.2%}")
    print(f"Recorded pipeline throughput: {float(overall['fps']):.2f} FPS")
    print("Validation uses supplied records only; it does not rerun training, inference or speed measurement.")


if __name__ == "__main__":
    main()
