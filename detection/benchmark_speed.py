"""Measure batch-1 end-to-end latency and FPS on CCPD images."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from common import (
    RESULTS_DIR,
    SEED,
    available_subsets,
    full_image_path,
    list_file_for_subset,
    optional_device,
    read_list,
    require_files,
)


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--subset", default="auto", help="auto selects the active dataset's first test subset")
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="")
    parser.add_argument("--output-name", default="ccpd_speed")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(weights)

    subset = available_subsets()[0] if args.subset.strip().lower() == "auto" else args.subset.strip().lower()
    if subset not in available_subsets():
        raise ValueError(f"Unknown subset {subset!r}; valid values: {available_subsets()}")
    relative_paths = read_list(list_file_for_subset(subset))
    rng = random.Random(SEED)
    rng.shuffle(relative_paths)
    requested = min(args.num_images + args.warmup, len(relative_paths))
    resolved_paths = [full_image_path(path) for path in relative_paths[:requested]]
    require_files(resolved_paths, f"{subset} benchmark image")
    paths = [str(path) for path in resolved_paths]
    warmup_paths = paths[: min(args.warmup, len(paths))]
    timed_paths = paths[len(warmup_paths):]
    if not timed_paths:
        raise ValueError("No images left for timed benchmark. Reduce --warmup.")

    device = optional_device(args.device)
    model = YOLO(str(weights))
    predict_args = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "verbose": False,
        "save": False,
    }
    if device is not None:
        predict_args["device"] = device

    print(f"Warmup on {len(warmup_paths)} images...")
    for path in warmup_paths:
        model.predict(source=path, **predict_args)
    sync_cuda()

    model_inference_times: list[float] = []
    print(f"Timing {len(timed_paths)} images with batch size 1...")
    sync_cuda()
    start = time.perf_counter()
    for index, path in enumerate(timed_paths, start=1):
        result = model.predict(source=path, **predict_args)[0]
        model_inference_times.append(float(result.speed.get("inference", np.nan)))
        if index % 100 == 0 or index == len(timed_paths):
            print(f"{index}/{len(timed_paths)}")
    sync_cuda()
    elapsed = time.perf_counter() - start

    end_to_end_ms = elapsed * 1000.0 / len(timed_paths)
    fps = len(timed_paths) / elapsed
    model_inference_ms = float(np.nanmean(model_inference_times))

    record = {
        "weights": str(weights),
        "subset": subset,
        "images": len(timed_paths),
        "warmup_images": len(warmup_paths),
        "imgsz": args.imgsz,
        "device_argument": args.device or "automatic",
        "end_to_end_ms_per_image": end_to_end_ms,
        "end_to_end_fps": fps,
        "ultralytics_mean_inference_ms": model_inference_ms,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{args.output_name}.csv"
    json_path = RESULTS_DIR / f"{args.output_name}.json"
    pd.DataFrame([record]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print("\nSpeed result:")
    print(f"Average end-to-end time: {end_to_end_ms:.2f} ms/image")
    print(f"End-to-end FPS: {fps:.2f}")
    print(f"Ultralytics model inference only: {model_inference_ms:.2f} ms/image")
    print("Saved:", csv_path)
    print("Saved:", json_path)


if __name__ == "__main__":
    main()
