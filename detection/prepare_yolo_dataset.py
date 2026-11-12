"""Convert CCPD2019 metadata or CCPD2020-Green folders to Ultralytics YOLO format."""
from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from common import (
    DATA_ROOT,
    DATASET_VARIANT,
    OUT_DIR,
    SEED,
    YOLO_DATASET_DIR,
    full_image_path,
    parse_filename,
    read_list,
    require_dataset,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def materialise_image(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "existing"

    methods = [mode] if mode != "auto" else ["hardlink", "symlink", "copy"]
    errors: list[str] = []
    for method in methods:
        try:
            if method == "hardlink":
                os.link(source, destination)
            elif method == "symlink":
                os.symlink(source, destination)
            elif method == "copy":
                shutil.copy2(source, destination)
            else:
                raise ValueError(f"Unsupported mode: {method}")
            return method
        except OSError as exc:
            errors.append(f"{method}: {exc}")

    raise OSError(f"Could not create {destination}. Attempts: {'; '.join(errors)}")


def yolo_line(row: pd.Series, image_width: int, image_height: int) -> tuple[str, bool]:
    original = (
        float(row["x_min"]),
        float(row["y_min"]),
        float(row["x_max"]),
        float(row["y_max"]),
    )
    x1 = min(max(original[0], 0.0), float(image_width))
    y1 = min(max(original[1], 0.0), float(image_height))
    x2 = min(max(original[2], 0.0), float(image_width))
    y2 = min(max(original[3], 0.0), float(image_height))
    clipped = (x1, y1, x2, y2) != original

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bounding box after clipping: {(x1, y1, x2, y2)}")

    centre_x = ((x1 + x2) / 2.0) / image_width
    centre_y = ((y1 + y2) / 2.0) / image_height
    box_width = (x2 - x1) / image_width
    box_height = (y2 - y1) / image_height
    values = [centre_x, centre_y, box_width, box_height]
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"YOLO values outside [0, 1]: {values}")
    return "0 " + " ".join(f"{value:.6f}" for value in values) + "\n", clipped


def green_metadata() -> pd.DataFrame:
    """Build metadata directly from the official CCPD-Green folder split."""
    rows: list[dict[str, object]] = []
    split_map = {"train": "train", "val": "validation", "test": "test"}
    for folder_name, split_name in split_map.items():
        folder = DATA_ROOT / folder_name
        paths = sorted(
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not paths:
            raise FileNotFoundError(f"No images found in {folder}")
        for path in paths:
            info = parse_filename(path.name)
            x1, y1, x2, y2 = info["bbox"]
            rows.append(
                {
                    "split": split_name,
                    "subset": "green",
                    "relative_path": path.relative_to(DATA_ROOT).as_posix(),
                    "x_min": x1,
                    "y_min": y1,
                    "x_max": x2,
                    "y_max": y2,
                }
            )
        print(f"Discovered {len(paths)} CCPD-Green {folder_name} images.")

    metadata = pd.DataFrame(rows)
    metadata.to_csv(OUT_DIR / "ccpd_green_metadata.csv", index=False, encoding="utf-8-sig")
    for split_name, list_name in (
        ("train", "train_green.txt"),
        ("validation", "val_green.txt"),
        ("test", "test_green.txt"),
    ):
        relative_paths = metadata.loc[metadata["split"] == split_name, "relative_path"].tolist()
        (OUT_DIR / list_name).write_text("\n".join(relative_paths) + "\n", encoding="utf-8")
    return metadata


def load_active_metadata() -> pd.DataFrame:
    require_dataset()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if DATASET_VARIANT == "ccpd2020_green":
        return green_metadata()

    metadata_path = OUT_DIR / "ccpd_selected_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing CCPD2019 metadata: {metadata_path}")
    return pd.read_csv(metadata_path)


def destination_parts(row: pd.Series, split_override: str | None = None) -> tuple[Path, Path]:
    relative = Path(str(row["relative_path"]).replace("\\", "/"))
    split = split_override or str(row["split"])
    subset = str(row["subset"])

    if split == "train":
        image_root = YOLO_DATASET_DIR / "images" / "train"
        label_root = YOLO_DATASET_DIR / "labels" / "train"
    elif split == "validation":
        image_root = YOLO_DATASET_DIR / "images" / "val"
        label_root = YOLO_DATASET_DIR / "labels" / "val"
    elif split == "test":
        image_root = YOLO_DATASET_DIR / "images" / "test" / subset
        label_root = YOLO_DATASET_DIR / "labels" / "test" / subset
    elif split == "dev":
        image_root = YOLO_DATASET_DIR / "images" / "dev"
        label_root = YOLO_DATASET_DIR / "labels" / "dev"
    else:
        raise ValueError(f"Unexpected split: {split}")

    # CCPD2019 uses subset folders such as ccpd_base. CCPD-Green already uses
    # train/val/test as split folders, so repeating that folder is unnecessary.
    parent_name = relative.parent.name
    subfolder = Path() if parent_name in {"train", "val", "test"} else Path(parent_name)
    return image_root / subfolder / relative.name, label_root / subfolder / f"{relative.stem}.txt"


def write_yaml(path: Path, dataset_root: Path, train: str, val: str, test: str | None = None) -> None:
    # The YAML file itself lives in dataset_root. Omitting an absolute `path:`
    # lets Ultralytics resolve train/val/test relative to this YAML, so the
    # whole project can be moved to another computer without editing paths.
    if path.parent.resolve() != dataset_root.resolve():
        raise ValueError(f"YAML must be stored in the dataset root: {dataset_root}")
    lines = [
        f'train: "{train}"',
        f'val: "{val}"',
    ]
    if test is not None:
        lines.append(f'test: "{test}"')
    lines.extend(["names:", "  0: license_plate", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def materialise_row(
    row: pd.Series,
    mode: str,
    force: bool,
    split_override: str | None = None,
) -> tuple[str, bool, Path, Path]:
    source = full_image_path(str(row["relative_path"]))
    image_path, label_path = destination_parts(row, split_override=split_override)
    if not source.is_file():
        raise FileNotFoundError(source)

    try:
        with Image.open(source) as image:
            width, height = image.size
    except OSError as exc:
        raise ValueError(f"Pillow could not read {source}: {exc}") from exc

    if force:
        image_path.unlink(missing_ok=True)
        label_path.unlink(missing_ok=True)
    method = materialise_image(source, image_path, mode)
    label_text, clipped = yolo_line(row, width, height)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(label_text, encoding="utf-8")
    return method, clipped, image_path, label_path


def build_dev_split(metadata: pd.DataFrame, mode: str, force: bool) -> tuple[int, int]:
    train_rows = metadata[metadata["split"] == "train"]
    if DATASET_VARIANT == "ccpd2020_green":
        dev_count = min(3000, len(train_rows))
        dev_rows = train_rows.sample(n=dev_count, random_state=SEED)
    else:
        dev_list_path = OUT_DIR / "train_dev_3k.txt"
        if not dev_list_path.exists():
            print("train_dev_3k.txt not found; skipping separate dev split.")
            return 0, 0
        dev_paths = {path.replace("\\", "/") for path in read_list(dev_list_path)}
        dev_rows = train_rows[
            train_rows["relative_path"].astype(str).str.replace("\\", "/", regex=False).isin(dev_paths)
        ]

    clipped_count = 0
    for index, (_, row) in enumerate(dev_rows.iterrows(), start=1):
        _, clipped, _, _ = materialise_row(row, mode, force, split_override="dev")
        clipped_count += int(clipped)
        if index % 500 == 0 or index == len(dev_rows):
            print(f"Dev split: {index}/{len(dev_rows)}")
    return len(dev_rows), clipped_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["auto", "hardlink", "symlink", "copy"],
        default="auto",
        help="auto tries hardlink, symlink, then copy.",
    )
    parser.add_argument("--force", action="store_true", help="Replace matching generated images and labels.")
    args = parser.parse_args()

    metadata = load_active_metadata()
    required = {"split", "subset", "relative_path", "x_min", "y_min", "x_max", "y_max"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    if metadata["relative_path"].duplicated().any():
        duplicates = metadata.loc[metadata["relative_path"].duplicated(), "relative_path"].head().tolist()
        raise ValueError(f"Duplicate relative paths in metadata: {duplicates}")

    YOLO_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    method_counts: dict[str, int] = {}
    clipped_count = 0

    for index, (_, row) in enumerate(metadata.iterrows(), start=1):
        method, clipped, image_path, label_path = materialise_row(row, args.mode, args.force)
        method_counts[method] = method_counts.get(method, 0) + 1
        clipped_count += int(clipped)
        manifest_rows.append(
            {
                "split": str(row["split"]),
                "subset": str(row["subset"]),
                "source_image": str(full_image_path(str(row["relative_path"]))),
                "yolo_image": str(image_path),
                "yolo_label": str(label_path),
                "materialisation": method,
                "bbox_clipped_to_image": str(bool(clipped)),
            }
        )
        if index % 1000 == 0 or index == len(metadata):
            print(f"Main dataset: {index}/{len(metadata)}")

    dev_count, dev_clipped = build_dev_split(metadata, args.mode, args.force)
    write_yaml(
        YOLO_DATASET_DIR / "ccpd_full.yaml",
        YOLO_DATASET_DIR,
        train="images/train",
        val="images/val",
        test="images/test",
    )
    if dev_count:
        write_yaml(
            YOLO_DATASET_DIR / "ccpd_dev.yaml",
            YOLO_DATASET_DIR,
            train="images/dev",
            val="images/val",
            test="images/test",
        )

    test_subsets = sorted(metadata.loc[metadata["split"] == "test", "subset"].unique())
    for subset in test_subsets:
        write_yaml(
            YOLO_DATASET_DIR / f"ccpd_test_{subset}.yaml",
            YOLO_DATASET_DIR,
            train="images/train",
            val="images/val",
            test=f"images/test/{subset}",
        )

    pd.DataFrame(manifest_rows).to_csv(
        YOLO_DATASET_DIR / "dataset_manifest.csv", index=False, encoding="utf-8-sig"
    )
    print("\nYOLO dataset preparation complete.")
    print("Dataset variant:", DATASET_VARIANT)
    print("DATA_ROOT:", DATA_ROOT)
    print("YOLO dataset:", YOLO_DATASET_DIR)
    print("Rows converted:", len(metadata))
    print("Dev rows:", dev_count)
    print("Bounding boxes clipped to image bounds:", clipped_count)
    print("Dev labels clipped to image bounds:", dev_clipped)
    print("Image methods:", method_counts)
    print("Random seed:", SEED)


if __name__ == "__main__":
    random.seed(SEED)
    main()
