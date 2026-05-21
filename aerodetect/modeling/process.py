"""Convert raw detection datasets into YOLO-ready layouts.

YOLO expects this directory structure::

    <dataset_root>/
        data.yaml
        images/
            train/  val/  test/
        labels/
            train/  val/  test/

Each label file mirrors the image filename and contains one row per box::

    <class_id> <x_center> <y_center> <width> <height>

with all coordinates normalised to ``[0, 1]``.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
import shutil

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from aerodetect.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


# Map CSV split labels to the directory names YOLO expects.
SPLIT_MAP: dict[str, str] = {
    "train": "train",
    "validation": "val",
    "val": "val",
    "test": "test",
}

YOLO_SPLITS: tuple[str, ...] = ("train", "val", "test")


class TransferMode(str, Enum):
    move = "move"
    copy = "copy"


def _ensure_yolo_dirs(root: Path) -> None:
    for split in YOLO_SPLITS:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def _to_yolo_bbox(
    xmin: float, ymin: float, xmax: float, ymax: float, width: int, height: int
) -> tuple[float, float, float, float]:
    """Convert an absolute (xmin, ymin, xmax, ymax) box to YOLO format."""

    # Clamp to the image bounds in case the annotations spill over.
    xmin = max(0.0, min(float(xmin), float(width)))
    xmax = max(0.0, min(float(xmax), float(width)))
    ymin = max(0.0, min(float(ymin), float(height)))
    ymax = max(0.0, min(float(ymax), float(height)))

    x_center = (xmin + xmax) / 2.0 / width
    y_center = (ymin + ymax) / 2.0 / height
    box_w = (xmax - xmin) / width
    box_h = (ymax - ymin) / height
    return x_center, y_center, box_w, box_h


def _write_data_yaml(root: Path, classes: list[str]) -> Path:
    yaml_path = root / "data.yaml"
    lines = [
        f"path: {root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(classes)}",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(classes))
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def process_military(
    labels_csv: Path | None = None,
    source_dir: Path | None = None,
    output_dir: Path | None = None,
    image_ext: str = ".jpg",
    transfer_mode: TransferMode = TransferMode.move,
    overwrite: bool = False,
) -> Path:
    """Convert the military aircraft dataset into a YOLO layout.

    Parameters
    ----------
    labels_csv:
        Path to ``labels_with_split.csv``. Defaults to the raw military folder.
    source_dir:
        Folder containing the raw ``<hash>.jpg`` images. Defaults to
        ``data/raw/military/dataset``.
    output_dir:
        Destination root. Defaults to ``data/processed/military``.
    image_ext:
        Image file extension to look for (case-insensitive).
    transfer_mode:
        ``move`` to relocate the images, ``copy`` to keep the source intact.
    overwrite:
        When ``True`` re-create labels and re-transfer images even if a label
        file already exists at the destination.

    Returns
    -------
    Path
        The dataset root containing ``images/``, ``labels/`` and ``data.yaml``.
    """

    military_raw = RAW_DATA_DIR / "military"
    labels_csv = labels_csv or military_raw / "labels_with_split.csv"
    source_dir = source_dir or military_raw / "dataset"
    output_dir = output_dir or PROCESSED_DATA_DIR / "military"

    if not labels_csv.is_file():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source image folder not found: {source_dir}")

    logger.info(f"Reading annotations from {labels_csv}")
    df = pd.read_csv(labels_csv)

    required_cols = {
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "split",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    unknown_splits = sorted(set(df["split"]) - SPLIT_MAP.keys())
    if unknown_splits:
        raise ValueError(f"Unknown split values in CSV: {unknown_splits}")

    classes = sorted(df["class"].unique())
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    logger.info(f"Found {len(classes)} classes and {df['filename'].nunique()} images")

    _ensure_yolo_dirs(output_dir)

    stats = {split: 0 for split in YOLO_SPLITS}
    missing_images: list[str] = []
    ext = image_ext if image_ext.startswith(".") else f".{image_ext}"

    grouped = df.groupby("filename", sort=False)
    for filename, rows in tqdm(grouped, total=grouped.ngroups, desc="military"):
        split = SPLIT_MAP[rows["split"].iloc[0]]
        label_path = output_dir / "labels" / split / f"{filename}.txt"
        image_dst = output_dir / "images" / split / f"{filename}{ext}"

        if label_path.exists() and image_dst.exists() and not overwrite:
            stats[split] += 1
            continue

        img_w = int(rows["width"].iloc[0])
        img_h = int(rows["height"].iloc[0])

        lines: list[str] = []
        box_iter = zip(
            rows["class"].to_numpy(),
            rows["xmin"].to_numpy(),
            rows["ymin"].to_numpy(),
            rows["xmax"].to_numpy(),
            rows["ymax"].to_numpy(),
        )
        for cls_name, xmin, ymin, xmax, ymax in box_iter:
            cls_id = class_to_id[cls_name]
            xc, yc, bw, bh = _to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h)
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        image_src = source_dir / f"{filename}{ext}"
        if not image_src.exists():
            missing_images.append(image_src.name)
            continue

        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        if image_dst.exists() and overwrite:
            image_dst.unlink()

        if not image_dst.exists():
            if transfer_mode is TransferMode.move:
                shutil.move(str(image_src), str(image_dst))
            else:
                shutil.copy2(str(image_src), str(image_dst))

        stats[split] += 1

    yaml_path = _write_data_yaml(output_dir, classes)

    logger.success(
        "military processed → "
        + ", ".join(f"{split}={stats[split]}" for split in YOLO_SPLITS)
    )
    logger.info(f"YOLO config written to {yaml_path}")
    if missing_images:
        logger.warning(
            f"{len(missing_images)} image(s) referenced in CSV but missing on disk; "
            f"first few: {missing_images[:5]}"
        )

    return output_dir


@app.command("military")
def cli_military(
    labels_csv: Path = typer.Option(
        None,
        "--labels-csv",
        help="Path to labels_with_split.csv (defaults to data/raw/military/labels_with_split.csv).",
    ),
    source_dir: Path = typer.Option(
        None,
        "--source-dir",
        help="Folder containing raw .jpg images (defaults to data/raw/military/dataset).",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        help="Destination root (defaults to data/processed/military).",
    ),
    image_ext: str = typer.Option(
        ".jpg", "--image-ext", help="Image extension to look for."
    ),
    transfer_mode: TransferMode = typer.Option(
        TransferMode.move,
        "--transfer-mode",
        case_sensitive=False,
        help="Move images from the source folder or copy them.",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Re-create labels and re-transfer existing images."
    ),
) -> None:
    """Convert the military aircraft dataset into a YOLO layout."""

    process_military(
        labels_csv=labels_csv,
        source_dir=source_dir,
        output_dir=output_dir,
        image_ext=image_ext,
        transfer_mode=transfer_mode,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    app()
