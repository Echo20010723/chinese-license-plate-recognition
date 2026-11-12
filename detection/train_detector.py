"""Train a YOLOv8 license-plate detector using the prepared CCPD dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from common import RUNS_DIR, SEED, YOLO_DATASET_DIR, optional_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(YOLO_DATASET_DIR / "ccpd_dev.yaml"))
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="", help="Examples: 0, 0,1, cpu, mps. Blank means automatic.")
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--name", default="ccpd_yolov8_dev")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path}. Run prepare_yolo_dataset.py first."
        )

    if args.resume and not Path(args.model).is_file():
        raise ValueError("--resume requires --model to point to an existing last.pt checkpoint.")

    model = YOLO(args.model)
    train_args = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(RUNS_DIR),
        "name": args.name,
        "exist_ok": True,
        "seed": SEED,
        "deterministic": True,
        "patience": args.patience,
        "close_mosaic": min(10, max(0, args.epochs // 5)),
        "plots": True,
        "save": True,
        "resume": args.resume,
    }

    device = optional_device(args.device)
    if device is not None:
        train_args["device"] = device
    if args.lr0 is not None:
        train_args["lr0"] = args.lr0

    print("Training configuration:")
    for key, value in train_args.items():
        print(f"  {key}: {value}")

    model.train(**train_args)

    trainer = getattr(model, "trainer", None)
    run_dir = Path(getattr(trainer, "save_dir", RUNS_DIR / args.name))
    print("\nTraining finished.")
    print("Run directory:", run_dir)
    print("Best weights:", run_dir / "weights" / "best.pt")
    print("Last weights:", run_dir / "weights" / "last.pt")


if __name__ == "__main__":
    main()
