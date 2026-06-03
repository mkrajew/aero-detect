"""Convert raw detection datasets into YOLO-ready layouts.

Processed datasets use this directory structure::

/
data.yaml
images/
  train/ val/ test/
labels/
  train/
    yolo/ rcnn/
  val/
    yolo/ rcnn/
  test/
    yolo/ rcnn/

Images are shared per split. YOLO labels are written as normalised ``.txt`` files,
while RCNN labels preserve the source annotation format per image.
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
import json
from pathlib import Path
import shutil

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from aerodetect.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

SPLIT_MAP: dict[str, str] = {
    "train": "train",
    "validation": "val",
    "val": "val",
    "test": "test",
}

YOLO_SPLITS: tuple[str, ...] = ("train", "val", "test")
LABEL_FORMATS: tuple[str, ...] = ("yolo", "rcnn")
SKYFUSION_SPLIT_MAP: dict[str, str] = {
    "train": "train",
    "val": "valid",
    "test": "test",
}


class TransferMode(str, Enum):
    move = "move"
    copy = "copy"


class ProcessDataset(str, Enum):
    military = "military"
    skyfusion = "skyfusion"
    all = "all"


def _ensure_dataset_dirs(root: Path) -> None:
    for split in YOLO_SPLITS:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        for label_format in LABEL_FORMATS:
            (root / "labels" / split / label_format).mkdir(parents=True, exist_ok=True)


def _to_yolo_bbox(
    xmin: float, ymin: float, xmax: float, ymax: float, width: int, height: int
) -> tuple[float, float, float, float]:
    """Convert an absolute (xmin, ymin, xmax, ymax) box to YOLO format."""
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


def _load_coco(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_fields = {"images", "annotations", "categories"}
    missing = required_fields - set(payload)
    if missing:
        raise ValueError(
            f"COCO annotations at {path} are missing required fields: {sorted(missing)}"
        )
    return payload


def process_military(
    source_dir: Path | None = None,
    output_dir: Path | None = None,
    transfer_mode: TransferMode = TransferMode.move,
    overwrite: bool = False,
) -> Path:
    """Convert the military aircraft dataset into a shared-image YOLO/RCNN layout."""
    military_raw = RAW_DATA_DIR / "military"
    labels_csv = military_raw / "labels_with_split.csv"
    source_dir = source_dir or military_raw / "dataset"
    output_dir = output_dir or PROCESSED_DATA_DIR / "military"
    ext = ".jpg"

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

    _ensure_dataset_dirs(output_dir)

    stats = {split: 0 for split in YOLO_SPLITS}
    missing_images: list[str] = []

    grouped = df.groupby("filename", sort=False)
    for filename, rows in tqdm(grouped, total=grouped.ngroups, desc="military"):
        split = SPLIT_MAP[rows["split"].iloc[0]]
        yolo_label_path = output_dir / "labels" / split / "yolo" / f"{filename}.txt"
        rcnn_label_path = output_dir / "labels" / split / "rcnn" / f"{filename}.csv"
        image_dst = output_dir / "images" / split / f"{filename}{ext}"

        if (
            yolo_label_path.exists()
            and rcnn_label_path.exists()
            and image_dst.exists()
            and not overwrite
        ):
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

        yolo_label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        rows.to_csv(rcnn_label_path, index=False)

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
        "military processed → " + ", ".join(f"{split}={stats[split]}" for split in YOLO_SPLITS)
    )
    logger.info(f"YOLO config written to {yaml_path}")
    if missing_images:
        logger.warning(
            f"{len(missing_images)} image(s) referenced in CSV but missing on disk; "
            f"first few: {missing_images[:5]}"
        )

    return output_dir


def process_skyfusion(
    source_dir: Path | None = None,
    output_dir: Path | None = None,
    transfer_mode: TransferMode = TransferMode.copy,
    overwrite: bool = False,
) -> Path:
    """Convert SkyFusion into a shared-image YOLO/RCNN layout."""
    source_dir = source_dir or RAW_DATA_DIR / "skyfusion" / "SkyFusion"
    output_dir = output_dir or PROCESSED_DATA_DIR / "skyfusion"

    if not source_dir.is_dir():
        raise FileNotFoundError(f"SkyFusion source folder not found: {source_dir}")

    split_dirs = {split: source_dir / name for split, name in SKYFUSION_SPLIT_MAP.items()}
    split_payloads: dict[str, dict] = {}
    for split, split_dir in split_dirs.items():
        if not split_dir.is_dir():
            raise FileNotFoundError(f"SkyFusion split folder not found: {split_dir}")

        ann_path = split_dir / "_annotations.coco.json"
        if not ann_path.is_file():
            raise FileNotFoundError(f"COCO annotations not found: {ann_path}")

        logger.info(f"Reading {split} annotations from {ann_path}")
        split_payloads[split] = _load_coco(ann_path)

    categories = sorted(split_payloads["train"]["categories"], key=lambda c: int(c["id"]))
    if not categories:
        raise ValueError("No categories found in SkyFusion annotations.")

    classes = [str(category["name"]) for category in categories]
    category_to_class_id = {int(category["id"]): idx for idx, category in enumerate(categories)}
    logger.info(f"Found {len(classes)} classes in SkyFusion")

    _ensure_dataset_dirs(output_dir)

    stats = {split: 0 for split in YOLO_SPLITS}
    missing_images: list[str] = []

    for split in YOLO_SPLITS:
        payload = split_payloads[split]
        split_dir = split_dirs[split]
        image_by_id = {int(image["id"]): image for image in payload["images"]}
        anns_by_image: dict[int, list[str]] = defaultdict(list)
        raw_anns_by_image: dict[int, list[dict]] = defaultdict(list)

        for ann in payload["annotations"]:
            image = image_by_id.get(int(ann["image_id"]))
            if image is None:
                continue

            raw_anns_by_image[int(ann["image_id"])] .append(ann)

            class_id = category_to_class_id.get(int(ann["category_id"]))
            if class_id is None:
                continue

            bbox = ann.get("bbox")
            if not isinstance(bbox, list) or len(bbox) < 4:
                continue

            x, y, w, h = map(float, bbox[:4])
            if w <= 0 or h <= 0:
                continue

            img_w = int(image["width"])
            img_h = int(image["height"])
            xc, yc, bw, bh = _to_yolo_bbox(x, y, x + w, y + h, img_w, img_h)
            if bw <= 0 or bh <= 0:
                continue

            anns_by_image[int(ann["image_id"])] .append(
                f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
            )

        for image in tqdm(payload["images"], desc=f"skyfusion/{split}"):
            image_id = int(image["id"])
            file_name = Path(str(image["file_name"])).name
            stem = Path(file_name).stem

            image_src = split_dir / file_name
            image_dst = output_dir / "images" / split / file_name
            yolo_label_path = output_dir / "labels" / split / "yolo" / f"{stem}.txt"
            rcnn_label_path = output_dir / "labels" / split / "rcnn" / f"{stem}.json"

            if (
                yolo_label_path.exists()
                and rcnn_label_path.exists()
                and image_dst.exists()
                and not overwrite
            ):
                stats[split] += 1
                continue

            if not image_src.exists():
                missing_images.append(str(image_src))
                continue

            lines = anns_by_image.get(image_id, [])
            yolo_label_path.write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )

            rcnn_payload = {
                "image": image,
                "annotations": raw_anns_by_image.get(image_id, []),
            }
            rcnn_label_path.write_text(
                json.dumps(rcnn_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

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
        "skyfusion processed → " + ", ".join(f"{split}={stats[split]}" for split in YOLO_SPLITS)
    )
    logger.info(f"YOLO config written to {yaml_path}")
    if missing_images:
        logger.warning(
            f"{len(missing_images)} image(s) referenced in COCO but missing on disk; "
            f"first few: {missing_images[:5]}"
        )

    return output_dir


@app.command()
def main(
    datasets: list[ProcessDataset] = typer.Argument(
        ...,
        case_sensitive=False,
        help=(
            "One or more datasets to process. "
            f"Available: {', '.join(d.value for d in ProcessDataset)}."
        ),
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Re-create labels and re-transfer existing images."
    ),
    military_source_dir: Path = typer.Option(
        RAW_DATA_DIR / "military" / "dataset",
        "--military-source-dir",
        help="Source folder with military images.",
    ),
    military_output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "military",
        "--military-output-dir",
        help="Output folder for processed military dataset.",
    ),
    military_transfer_mode: TransferMode = typer.Option(
        TransferMode.move,
        "--military-transfer-mode",
        case_sensitive=False,
        help="Move or copy military images.",
    ),
    skyfusion_source_dir: Path = typer.Option(
        RAW_DATA_DIR / "skyfusion" / "SkyFusion",
        "--skyfusion-source-dir",
        help="Source folder with SkyFusion train/valid/test splits.",
    ),
    skyfusion_output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "skyfusion",
        "--skyfusion-output-dir",
        help="Output folder for processed SkyFusion dataset.",
    ),
    skyfusion_transfer_mode: TransferMode = typer.Option(
        TransferMode.copy,
        "--skyfusion-transfer-mode",
        case_sensitive=False,
        help="Move or copy SkyFusion images.",
    ),
) -> None:
    """Convert selected raw datasets into shared-image YOLO/RCNN layouts."""
    if any(d == ProcessDataset.all for d in datasets):
        targets = [ProcessDataset.military, ProcessDataset.skyfusion]
    else:
        targets = datasets

    for dataset in targets:
        if dataset == ProcessDataset.military:
            path = process_military(
                source_dir=military_source_dir,
                output_dir=military_output_dir,
                transfer_mode=military_transfer_mode,
                overwrite=overwrite,
            )
        elif dataset == ProcessDataset.skyfusion:
            path = process_skyfusion(
                source_dir=skyfusion_source_dir,
                output_dir=skyfusion_output_dir,
                transfer_mode=skyfusion_transfer_mode,
                overwrite=overwrite,
            )
        else:
            raise ValueError(f"Unsupported dataset value: {dataset}")

        logger.info(f"[{dataset.value}] processed dataset path: {path}")


if __name__ == "__main__":
    app()
