from enum import Enum
from pathlib import Path

import kagglehub
from loguru import logger
import typer

from aerodetect.config import RAW_DATA_DIR

app = typer.Typer()


class DatasetName(str, Enum):
    military = "military"
    skyfusion = "skyfusion"
    all = "all"


DATASETS: dict[str, str] = {
    "military": "a2015003713/militaryaircraftdetectiondataset",
    "skyfusion": "kailaspsudheer/tiny-object-detection",
}


def is_dataset_downloaded(target_dir: Path) -> bool:
    return target_dir.is_dir() and any(target_dir.iterdir())


def download_dataset(
    name: str, slug: str, output_dir: Path, force: bool = False
) -> Path:
    target_dir = output_dir / name
    if not force and is_dataset_downloaded(target_dir):
        logger.info(
            f"{name} dataset already present at {target_dir}, skipping download."
        )
        return target_dir

    logger.info(f"Downloading {name} dataset...")
    path = Path(kagglehub.dataset_download(slug, output_dir=target_dir))
    logger.success(f"Downloaded {name} dataset to {path}")
    return path


@app.command()
def main(
    datasets: list[DatasetName] = typer.Argument(
        ...,
        case_sensitive=False,
        help=(
            "One or more datasets to download. "
            f"Available: {', '.join(d.value for d in DatasetName)}."
        ),
    ),
    output_dir: Path = typer.Option(
        RAW_DATA_DIR,
        "--output-dir",
        "-o",
        help="Directory used as a reference for raw data (datasets are cached by kagglehub).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-download datasets even if they already exist locally.",
    ),
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if any(d == DatasetName.all for d in datasets):
        targets = list(DATASETS.keys())
    else:
        targets = [d.value for d in datasets]

    for name in targets:
        path = download_dataset(name, DATASETS[name], output_dir, force=force)
        logger.info(f"[{name}] dataset path: {path}")


if __name__ == "__main__":
    app()
