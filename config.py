"""Portable project configuration for CCPD detection and recognition."""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def _is_green_dataset(path: Path) -> bool:
    """Return True for the official CCPD2020-Green train/val/test layout."""
    return all((path / split).is_dir() for split in ("train", "val", "test"))


def _is_ccpd2019_dataset(path: Path) -> bool:
    """Keep compatibility with the original CCPD2019 project layout."""
    return (path / "splits").is_dir() or any(path.glob("ccpd_*"))


# Teammates normally only need to put the data beside this file as shown below.
# CCPD_DATA_ROOT is an optional override when the dataset is stored elsewhere.
override = os.environ.get("CCPD_DATA_ROOT")
DATA_ROOTS = [
    Path(override).expanduser() if override else None,
    PROJECT_DIR / "CCPD2020" / "ccpd_green",
    PROJECT_DIR / "ccpd_green",
    PROJECT_DIR / "CCPD2019",
    PROJECT_DIR.parent / "CCPD2019",
]

DATA_ROOT = next(
    (
        path.resolve()
        for path in DATA_ROOTS
        if path is not None and path.is_dir() and (_is_green_dataset(path) or _is_ccpd2019_dataset(path))
    ),
    None,
)

# Configuration and single-image inference must work without the training data.
# An explicit override must never silently fall back to another dataset.
if override:
    DATA_ROOT = Path(override).expanduser().resolve()
elif DATA_ROOT is None:
    DATA_ROOT = PROJECT_DIR / "CCPD2019"


def require_dataset() -> Path:
    """Validate the dataset only when a dataset operation is requested."""
    if not DATA_ROOT.is_dir() or not (
        _is_green_dataset(DATA_ROOT) or _is_ccpd2019_dataset(DATA_ROOT)
    ):
        raise FileNotFoundError(
            f"CCPD dataset not found at {DATA_ROOT}. Set CCPD_DATA_ROOT to "
            "CCPD2019 (ccpd_* folders) or CCPD2020/ccpd_green (train/val/test)."
        )
    return DATA_ROOT

DATASET_VARIANT = "ccpd2020_green" if _is_green_dataset(DATA_ROOT) else "ccpd2019"
SPLIT_DIR = DATA_ROOT / "splits"
OUT_DIR = PROJECT_DIR / "data_preparation"

SEED = 9444
N_TRAIN = 30000
N_VAL = 5000
