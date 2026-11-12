"""Evaluate one-box CCPD detection accuracy and optional Ultralytics AP metrics."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from common import (
    RESULTS_DIR,
    YOLO_DATASET_DIR,
    full_image_path,
    list_file_for_subset,
    optional_device,
    parse_filename,
    parse_subsets,
    read_list,
    require_files,
)


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def evaluate_ccpd_top1(
    model: YOLO,
    subset: str,
    imgsz: int,
    conf: float,
    batch: int,
    device,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    relative_paths = read_list(list_file_for_subset(subset))
    resolved_paths = [full_image_path(path) for path in relative_paths]
    require_files(resolved_paths, f"{subset} evaluation image")
    records: list[dict[str, object]] = []
    correct = 0
    detected = 0
    ious_all: list[float] = []
    ious_detected: list[float] = []
    inference_ms: list[float] = []

    # A Python list is treated by recent Ultralytics versions as one in-memory
    # batch. Passing all 3,000 paths at once therefore exhausts an 8 GB GPU.
    # Predict in explicit small chunks while preserving path/result alignment.
    prediction_batch = max(1, min(int(batch), 16))
    total = len(relative_paths)
    for start in range(0, total, prediction_batch):
        stop = min(start + prediction_batch, total)
        batch_relative_paths = relative_paths[start:stop]
        batch_image_paths = [str(path) for path in resolved_paths[start:stop]]
        predict_args = {
            "source": batch_image_paths,
            "stream": False,
            "batch": prediction_batch,
            "imgsz": imgsz,
            "conf": conf,
            "verbose": False,
            "save": False,
        }
        if device is not None:
            predict_args["device"] = device

        batch_results = model.predict(**predict_args)
        if len(batch_results) != len(batch_relative_paths):
            raise RuntimeError(
                f"Ultralytics returned {len(batch_results)} results for "
                f"{len(batch_relative_paths)} input images."
            )

        for relative_path, result in zip(batch_relative_paths, batch_results):
            gt_box = np.asarray(parse_filename(relative_path)["bbox"], dtype=float)
            pred_box = None
            best_confidence = 0.0
            iou = 0.0

            if result.boxes is not None and len(result.boxes) > 0:
                confidences = result.boxes.conf.detach().cpu().numpy()
                best_index = int(np.argmax(confidences))
                best_confidence = float(confidences[best_index])
                pred_box = result.boxes.xyxy[best_index].detach().cpu().numpy().astype(float)
                iou = box_iou(gt_box, pred_box)
                detected += 1
                ious_detected.append(iou)

            is_correct = iou > 0.7
            correct += int(is_correct)
            ious_all.append(iou)
            inference_ms.append(float(result.speed.get("inference", np.nan)))

            records.append(
                {
                    "subset": subset,
                    "relative_path": relative_path,
                    "gt_x1": gt_box[0],
                    "gt_y1": gt_box[1],
                    "gt_x2": gt_box[2],
                    "gt_y2": gt_box[3],
                    "pred_x1": np.nan if pred_box is None else pred_box[0],
                    "pred_y1": np.nan if pred_box is None else pred_box[1],
                    "pred_x2": np.nan if pred_box is None else pred_box[2],
                    "pred_y2": np.nan if pred_box is None else pred_box[3],
                    "confidence": best_confidence,
                    "iou": iou,
                    "correct_iou_gt_0_7": is_correct,
                }
            )

        if stop % 200 == 0 or stop == total:
            print(f"{subset}: {stop}/{total}")

    if len(records) != total:
        raise RuntimeError(
            f"Ultralytics returned {len(records)} results for {total} {subset} images; "
            "evaluation would be incomplete."
        )
    summary: dict[str, float | int | str] = {
        "subset": subset,
        "images": total,
        "detections": detected,
        "ccpd_correct_iou_gt_0_7": correct,
        "ccpd_accuracy_iou_gt_0_7": correct / total if total else 0.0,
        "mean_top1_iou": float(np.mean(ious_all)) if ious_all else 0.0,
        "mean_iou_detected_only": float(np.mean(ious_detected)) if ious_detected else 0.0,
        "mean_model_inference_ms": float(np.nanmean(inference_ms)) if inference_ms else np.nan,
    }
    return summary, pd.DataFrame(records)


def add_ultralytics_metrics(
    model: YOLO,
    summary: dict[str, float | int | str],
    subset: str,
    imgsz: int,
    batch: int,
    device,
) -> None:
    yaml_path = YOLO_DATASET_DIR / f"ccpd_test_{subset}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing test YAML: {yaml_path}")

    val_args = {
        "data": str(yaml_path),
        "split": "test",
        "imgsz": imgsz,
        "batch": batch,
        "plots": False,
        "verbose": False,
    }
    if device is not None:
        val_args["device"] = device

    metrics = model.val(**val_args)
    summary["precision"] = float(metrics.box.mp)
    summary["recall"] = float(metrics.box.mr)
    summary["mAP50"] = float(metrics.box.map50)
    summary["mAP75"] = float(getattr(metrics.box, "map75", np.nan))
    summary["mAP50_95"] = float(metrics.box.map)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--subsets",
        default="auto",
        help="auto/all selects Green for CCPD2020, or all configured CCPD2019 subsets.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--with-map",
        action="store_true",
        help="Also run Ultralytics validation for Precision/Recall/mAP. This adds a second inference pass.",
    )
    parser.add_argument("--output-name", default="ccpd_detection_evaluation")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(weights)

    device = optional_device(args.device)
    subsets = parse_subsets(args.subsets)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / f"{args.output_name}_summary.csv"
    details_path = RESULTS_DIR / f"{args.output_name}_per_image.csv"

    model = YOLO(str(weights))
    summaries: list[dict[str, float | int | str]] = []
    all_details: list[pd.DataFrame] = []

    for subset in subsets:
        print(f"\nEvaluating subset: {subset}")
        summary, details = evaluate_ccpd_top1(
            model=model,
            subset=subset,
            imgsz=args.imgsz,
            conf=args.conf,
            batch=args.batch,
            device=device,
        )
        if args.with_map:
            add_ultralytics_metrics(
                model=model,
                summary=summary,
                subset=subset,
                imgsz=args.imgsz,
                batch=args.batch,
                device=device,
            )
        summaries.append(summary)
        all_details.append(details)
        # Checkpoint after every subset so a later failure never discards
        # already-completed evaluation work.
        pd.DataFrame(summaries).to_csv(summary_path, index=False, encoding="utf-8-sig")
        pd.concat(all_details, ignore_index=True).to_csv(
            details_path, index=False, encoding="utf-8-sig"
        )
        print(f"Checkpoint saved after {subset}: {summary_path}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(summaries)
    details_df = pd.concat(all_details, ignore_index=True)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")

    display = summary_df.copy()
    for column in [
        "ccpd_accuracy_iou_gt_0_7", "mean_top1_iou", "mean_iou_detected_only",
        "precision", "recall", "mAP50", "mAP75", "mAP50_95"
    ]:
        if column in display.columns:
            display[column] = display[column].map(lambda value: f"{value:.4f}")

    print("\nEvaluation summary:")
    print(display.to_string(index=False))
    print("\nSaved:", summary_path)
    print("Saved:", details_path)


if __name__ == "__main__":
    main()
