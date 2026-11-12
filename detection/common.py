"""Shared paths and helpers for license plate detection."""
from __future__ import annotations

import os
import sys
from pathlib import Path


PART1_DIR = Path(__file__).resolve().parent.parent
if str(PART1_DIR) not in sys.path:
    sys.path.insert(0, str(PART1_DIR))

# Do not wrap configuration errors: if a teammate has not placed the dataset in
# the expected folder, config.py now reports the exact searched paths directly.
from config import DATA_ROOT, DATASET_VARIANT, OUT_DIR, SEED, require_dataset  # type: ignore
from ccpd_annotations import parse_filename  # type: ignore

DATA_ROOT = Path(DATA_ROOT)
OUT_DIR = Path(OUT_DIR)
SEED = int(SEED)
DATASET_VARIANT = str(DATASET_VARIANT)
PART2_DIR = Path(__file__).resolve().parent

# Keep generated YOLO folders for CCPD2019 and CCPD2020-Green separate so stale
# files from one dataset can never leak into an experiment on the other.
YOLO_DATASET_DIR = PART2_DIR / (
    "yolo_dataset_green" if DATASET_VARIANT == "ccpd2020_green" else "yolo_dataset"
)
RUNS_DIR = PART2_DIR / "runs"
RESULTS_DIR = PART2_DIR / "results"


def read_list(path: os.PathLike[str] | str) -> list[str]:
    with open(path, "r", encoding="utf-8-sig") as file:
        return [line.strip() for line in file if line.strip()]


def full_image_path(relative_path: str) -> Path:
    require_dataset()
    clean = relative_path.replace("\\", "/")
    return DATA_ROOT / Path(clean)


def list_file_for_subset(subset: str) -> Path:
    if subset == "green":
        return OUT_DIR / "test_green.txt"
    if subset == "base":
        return OUT_DIR / "test_base_3k.txt"
    return OUT_DIR / f"test_{subset}_1k.txt"


def available_subsets() -> list[str]:
    if DATASET_VARIANT == "ccpd2020_green":
        return ["green"]
    return ["base", "blur", "challenge", "db", "fn", "rotate", "tilt", "weather"]


def parse_subsets(text: str) -> list[str]:
    allowed = available_subsets()
    if text.strip().lower() in {"all", "auto"}:
        return allowed
    requested = [item.strip().lower() for item in text.split(",") if item.strip()]
    if not requested:
        raise ValueError("Specify at least one subset, or 'all'.")
    invalid = [item for item in requested if item not in allowed]
    if invalid:
        raise ValueError(f"Unknown subset(s): {invalid}. Valid values for {DATASET_VARIANT}: {allowed}")
    return requested


def require_files(paths: list[Path], description: str) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        preview = "\n  - ".join(str(path) for path in missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} {description} file(s). First missing paths:\n  - {preview}"
        )


def optional_device(value: str | None):
    if value is None or str(value).strip() == "":
        return None
    return value
