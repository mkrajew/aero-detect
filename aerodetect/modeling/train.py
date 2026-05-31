"""Train a YOLO model with Hydra configuration.

When running this module, you must specify the Hydra `db` group, for example:
`uv run ./aerodetec/modeling/train.py +db=military`.
"""

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO

from aerodetect.config import MODELING_DIR, PROCESSED_DATA_DIR


@hydra.main(
    version_base=None,
    config_path=str(MODELING_DIR / "conf"),
    config_name="config",
)
def train(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    model = YOLO(cfg.db.model)
    logger.info(f"Loaded {cfg.db.model} model.")
    results = model.train(
        data=PROCESSED_DATA_DIR / cfg.db.dataset,
        epochs=cfg.db.epochs,
        imgsz=cfg.db.imgsz,
        device=cfg.db.device,
        patience=cfg.db.patience,
        batch=cfg.db.batch,
        project=cfg.wandb.project,
        name=cfg.db.run,
        seed=cfg.db.seed,
        workers=cfg.db.workers,
        # Augmentations
        scale=cfg.aug.scale,
        degrees=cfg.aug.degrees,
        translate=cfg.aug.transalate,
        shear=cfg.aug.shear,
        perspective=cfg.aug.perspective,
        fliplr=cfg.aug.fliplr,
        flipud=cfg.aug.flipud,
        hsv_h=cfg.aug.hsv_h,
        hsv_s=cfg.aug.hsv_s,
        hsv_v=cfg.aug.hsv_v,
    )

    metrics = model.val(
        device=cfg.db.device,
        project=cfg.wandb.project,
        name=f"{cfg.db.run}-val",
    )

    return results, metrics


if __name__ == "__main__":
    train()
