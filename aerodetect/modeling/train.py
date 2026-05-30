"""Train a YOLO model with Hydra configuration.

When running this module, you must specify the Hydra `db` group, for example:
`uv run ./aerodetec/modeling/train.py +db=military`.
"""

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO
import wandb

from aerodetect.config import MODELING_DIR, PROCESSED_DATA_DIR


@hydra.main(
    version_base=None,
    config_path=str(MODELING_DIR / "conf"),
    config_name="config",
)
def train(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved_cfg, dict):
        resolved_cfg = {"config": resolved_cfg}

    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.db.run,
        config={
            **resolved_cfg,
            "resolved_data_path": str(PROCESSED_DATA_DIR / cfg.db.dataset),
        },
    )

    model = YOLO(cfg.db.model)
    logger.info(f"Loaded {cfg.db.model} model.")
    results = model.train(
        data=PROCESSED_DATA_DIR / cfg.db.dataset,
        epochs=cfg.db.epochs,
        imgsz=cfg.db.imgsz,
        device=cfg.db.device,
        patience=cfg.db.patience,
        batch=cfg.db.batch,
        project=cfg.db.project,
        name=cfg.db.run,
        seed=cfg.db.seed,
        workers=cfg.db.workers,
    )

    metrics = model.val()

    wandb.log(
        {
            "final/mAP50-95": metrics.box.map,
            "final/mAP50": metrics.box.map50,
            "final/precision": metrics.box.mp,
            "final/recall": metrics.box.mr,
        }
    )
    wandb.finish()
    return results, metrics


if __name__ == "__main__":
    train()
