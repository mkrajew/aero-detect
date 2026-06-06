"""Evaluate a YOLO checkpoint on a processed test dataset."""

from enum import Enum
from pathlib import Path

from loguru import logger
import pandas as pd
import typer
from ultralytics import YOLO

from aerodetect.config import PROCESSED_DATA_DIR, REPORTS_DIR

app = typer.Typer(add_completion=False)


class Dataset(str, Enum):
    military = "military"
    skyfusion = "skyfusion"


@app.command()
def main(
    checkpoint: Path,
    dataset: Dataset,
    experiment: str = "3-aug",
    device: str = "0",
    batch: int = 16,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.5,
):
    """Calculate accuracy, error and inference-speed statistics."""

    output_dir = (
        REPORTS_DIR / "evaluation" / f"{experiment}-{dataset.value}"
    ).resolve()
    results = YOLO(checkpoint).val(
        data=PROCESSED_DATA_DIR / dataset.value / "data.yaml",
        split="test",
        device=device,
        batch=batch,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        workers=0,
        plots=True,
        project=output_dir.parent,
        name=output_dir.name,
        exist_ok=True,
    )

    precision, recall, map50, map50_95 = map(float, results.box.mean_results())
    matrix = results.confusion_matrix.matrix
    true_positives = matrix.diagonal()[:-1]
    false_positives = matrix.sum(axis=1)[:-1] - true_positives
    false_negatives = matrix.sum(axis=0)[:-1] - true_positives

    class_metrics = pd.DataFrame(results.summary())
    class_ids = results.box.ap_class_index.astype(int)
    class_metrics.insert(0, "class_id", class_ids)
    class_metrics["TP"] = true_positives[class_ids]
    class_metrics["FP"] = false_positives[class_ids]
    class_metrics["FN"] = false_negatives[class_ids]
    class_metrics["efficiency_score"] = (
        class_metrics["mAP50-95"] * 1000 / results.speed["inference"]
    )

    total_ms = sum(
        results.speed[key] for key in ("preprocess", "inference", "postprocess")
    )
    summary = pd.DataFrame(
        [
            {
                "experiment": experiment,
                "dataset": dataset.value,
                "checkpoint": str(checkpoint),
                "images": len(
                    list((PROCESSED_DATA_DIR / dataset.value / "images/test").iterdir())
                ),
                "instances": int(results.nt_per_class.sum()),
                "precision": precision,
                "recall": recall,
                "f1": (
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else 0
                ),
                "mAP50": map50,
                "mAP50-95": map50_95,
                "TP": int(true_positives.sum()),
                "FP": int(false_positives.sum()),
                "FN": int(false_negatives.sum()),
                "preprocess_ms": results.speed["preprocess"],
                "inference_ms": results.speed["inference"],
                "postprocess_ms": results.speed["postprocess"],
                "total_ms": total_ms,
                "fps": 1000 / total_ms,
                "efficiency_score": map50_95 * 1000 / results.speed["inference"],
            }
        ]
    )

    summary.to_csv(output_dir / "summary.csv", index=False)
    class_metrics.to_csv(output_dir / "class_metrics.csv", index=False)
    logger.success(f"Results saved to {output_dir}")


if __name__ == "__main__":
    app()
