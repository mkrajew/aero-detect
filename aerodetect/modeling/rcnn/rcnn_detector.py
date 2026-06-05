from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import wandb

from military_dataset import MilitaryDataset
from aerodetect.config import RCNN_CHECKPOINTS_DIR


MODEL_REGISTRY = {
    "fasterrcnn_mobilenet_v3_large_fpn": fasterrcnn_mobilenet_v3_large_fpn,
    "fasterrcnn_resnet50_fpn_v2": fasterrcnn_resnet50_fpn_v2,
    "fasterrcnn_resnet50_fpn": fasterrcnn_resnet50_fpn,
}

DATASET_REGISTRY = {
    "military": MilitaryDataset,
    "skyfusion": None,
}


def collate_fn(batch):
    return tuple(zip(*batch))


class RcnnDetector:
    def __init__(
        self,
        model_name,
        dataset_name,
        lr=0.001,
        epochs=12,
        batchsize=2,
        num_workers=1,
        augs=False,
        trainable_backbone_layers=3,
        pretrained=True,
        weighted_sampling=False,
        img_size=640,
        warmup_epochs=0,
        momentum=0.9,
        weight_decay=0.0005,
        optimizer_name="sgd",
        scheduler_name="none",   # none | step | cosine | warmup_cosine
        scheduler_step_size=None,
        scheduler_gamma=0.1,
        use_amp=True,
        class_metrics=True,
        eval_score_threshold=0.5,
        log_prediction_batches=(1, 2, 4, 5),
        subset_train_size=None,
        subset_val_size=None,
        shuffle_train=True,
        num_classes=None,
        save_best_metric="map",
        run_name = None,
        sweep_name = None
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self._model_ref = MODEL_REGISTRY[model_name]
        self._dataset_ref = DATASET_REGISTRY[dataset_name]

        self.lr = lr
        self.epochs = epochs
        self.batchsize = batchsize
        self.num_workers = num_workers
        self.augs = augs
        self.trainable_backbone_layers = trainable_backbone_layers
        self.pretrained = pretrained
        self.weighted_sampling = weighted_sampling
        self.img_size = img_size
        self.warmup_epochs = warmup_epochs
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.use_amp = use_amp
        self.class_metrics = class_metrics
        self.eval_score_threshold = eval_score_threshold
        self.log_prediction_batches = set(log_prediction_batches)
        self.subset_train_size = subset_train_size
        self.subset_val_size = subset_val_size
        self.shuffle_train = shuffle_train
        self.save_best_metric = save_best_metric
        self.run_name = run_name
        self.sweep_name = sweep_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes if num_classes is not None else self._dataset_ref.get_class_number()

        self.scaler = None
        self.build_model()

    def build_dataset(self, split):
        return self._dataset_ref(split=split, img_size=self.img_size, augment=(self.augs and split == "train"))

    def build_training_data(self):
        self.train_dataset = self.build_dataset("train")
        self.val_dataset = self.build_dataset("val")

        train_dataset = self.train_dataset
        val_dataset = self.val_dataset

        if self.subset_train_size is not None:
            train_dataset = Subset(train_dataset, range(min(self.subset_train_size, len(train_dataset))))
        if self.subset_val_size is not None:
            val_dataset = Subset(val_dataset, range(min(self.subset_val_size, len(val_dataset))))

        train_sampler = None
        train_shuffle = self.shuffle_train

        if self.weighted_sampling:
            if isinstance(train_dataset, Subset):
                base_dataset = train_dataset.dataset
                base_weights = torch.tensor(base_dataset.build_sampling_weight_map(), dtype=torch.double)
                subset_indices = train_dataset.indices
                dataset_weights = base_weights[subset_indices]
            else:
                dataset_weights = torch.tensor(train_dataset.build_sampling_weight_map(), dtype=torch.double)

            train_sampler = WeightedRandomSampler(
                weights=dataset_weights,
                num_samples=len(dataset_weights),
                replacement=True,
            )
            train_shuffle = False

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batchsize,
            shuffle=train_shuffle if train_sampler is None else False,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batchsize,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        return train_loader, val_loader

    def build_test_data(self):
        self.test_dataset = self.build_dataset("test")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batchsize,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def build_model(self):
        if self.pretrained:
            model = self._model_ref(
                weights="DEFAULT",
                trainable_backbone_layers=self.trainable_backbone_layers,
            )
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        else:
            model = self._model_ref(
                num_classes=self.num_classes,
                trainable_backbone_layers=self.trainable_backbone_layers,
            )

        model.to(self.device)
        self.model = model

    def build_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]

        if self.optimizer_name.lower() == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name.lower() == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer_name: {self.optimizer_name}")

    def build_scheduler(self, optimizer):
        if self.scheduler_name == "none":
            return None

        if self.scheduler_name == "step":
            step_size = self.scheduler_step_size or max(1, self.epochs // 3)
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=self.scheduler_gamma,
            )

        if self.scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.epochs,
            )

        if self.scheduler_name == "warmup_cosine":
            if self.warmup_epochs <= 0:
                raise ValueError("warmup_epochs must be > 0 for warmup_cosine")
            scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=self.warmup_epochs,
            )
            scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.epochs - self.warmup_epochs,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scheduler_warmup, scheduler_main],
                milestones=[self.warmup_epochs],
            )

        raise ValueError(f"Unsupported scheduler_name: {self.scheduler_name}")

    def _move_batch_to_device(self, images, targets):
        images = [img.to(self.device) for img in images]
        targets = [
            {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]
        return images, targets

    def train_one_epoch(self, data_loader, optimizer, epoch_id, run):
        running_losses = defaultdict(float)
        epoch_losses = defaultdict(float)
        self.model.train()

        amp_enabled = self.use_amp and self.device.type == "cuda"
        if self.scaler is None:
            self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        for i, (images, targets) in enumerate(tqdm(data_loader)):
            optimizer.zero_grad(set_to_none=True)
            images, targets = self._move_batch_to_device(images, targets)

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())

            for k, v in loss_dict.items():
                val = v.detach().item()
                running_losses[k] += val
                epoch_losses[k] += val

            running_losses["loss"] += loss.detach().item()
            epoch_losses["loss"] += loss.detach().item()

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

        num_batches = len(data_loader)
        epoch_avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}

        print(
            f"Epoch {epoch_id + 1} summary | " +
            " | ".join(f"{k}: {v:.4f}" for k, v in epoch_avg_losses.items())
        )

        run.log({
            "epoch": epoch_id + 1,
            "train/lr": optimizer.param_groups[0]["lr"],
            **{f"train/epoch/{k}": v for k, v in epoch_avg_losses.items()},
        })

        return epoch_avg_losses["loss"]

    def eval(self, data_loader, metric, run, epoch):
        self.model.eval()
        metric.reset()

        with torch.no_grad():
            for i, (images, targets) in enumerate(tqdm(data_loader)):
                images, targets = self._move_batch_to_device(images, targets)
                outputs = self.model(images)

                outputs_cpu = [{k: v.cpu() for k, v in out.items()} for out in outputs]
                targets_cpu = [{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

                metric.update(outputs_cpu, targets_cpu)

                if i in self.log_prediction_batches:
                    img = (images[0].cpu() * 255).to(torch.uint8)

                    pred_boxes = outputs_cpu[0]["boxes"]
                    pred_labels = outputs_cpu[0]["labels"]
                    pred_scores = outputs_cpu[0]["scores"]

                    gt_boxes = targets_cpu[0]["boxes"]
                    gt_labels = targets_cpu[0]["labels"]

                    if hasattr(self.val_dataset, "idx_to_class"):
                        idx_to_class = self.val_dataset.idx_to_class
                    elif isinstance(self.val_dataset, Subset) and hasattr(self.val_dataset.dataset, "idx_to_class"):
                        idx_to_class = self.val_dataset.dataset.idx_to_class
                    else:
                        idx_to_class = {}

                    pred_box_labels = [
                        f"PRED {idx_to_class.get(l.item(), str(l.item()))} {s:.2f}"
                        for l, s in zip(pred_labels, pred_scores)
                    ]

                    gt_box_labels = [
                        f"GT {idx_to_class.get(l.item(), str(l.item()))}"
                        for l in gt_labels
                    ]

                    img_with_gt = draw_bounding_boxes(
                        img,
                        gt_boxes,
                        labels=gt_box_labels,
                        colors="green",
                        width=3,
                    )

                    img_with_boxes = draw_bounding_boxes(
                        img_with_gt,
                        pred_boxes,
                        labels=pred_box_labels,
                        colors="red",
                        width=2,
                    )

                    try:
                        run.log({
                            "epoch": epoch + 1,
                            f"val/predictions_vs_gt/batch_{i}": wandb.Image(
                                to_pil_image(img_with_boxes),
                                caption=f"epoch={epoch + 1}, batch_index={i}"
                            ),
                        })
                    except FileNotFoundError as e:
                        print(f"[WARN] Skipping wandb image log at epoch={epoch}, i={i}: {e}")

        results = metric.compute()
        print(results)
        return results

    def train(self):
        best_metric_value = float("-inf")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        runname = f"{self.run_name}_{timestamp}" if self.run_name else f"{self.model_name}_{timestamp}"

        run = wandb.init(
            project="detection-yolo",
            name=runname,
            reinit= True,
            group = self.sweep_name ,
            config={
                "model_name": self.model_name,
                "dataset_name": self.dataset_name,
                "num_classes": self.num_classes,
                "lr": self.lr,
                "epochs": self.epochs,
                "batchsize": self.batchsize,
                "img_size": self.img_size,
                "augs": self.augs,
                "trainable_backbone_layers": self.trainable_backbone_layers,
                "weighted_sampling": self.weighted_sampling,
                "pretrained": self.pretrained,
                "warmup_epochs": self.warmup_epochs,
                "momentum": self.momentum,
                "weight_decay": self.weight_decay,
                "optimizer_name": self.optimizer_name,
                "scheduler_name": self.scheduler_name,
                "scheduler_step_size": self.scheduler_step_size,
                "scheduler_gamma": self.scheduler_gamma,
                "use_amp": self.use_amp,
                "class_metrics": self.class_metrics,
                "subset_train_size": self.subset_train_size,
                "subset_val_size": self.subset_val_size,
                "device": str(self.device),
            },
        )

        train_loader, val_loader = self.build_training_data()
        optimizer = self.build_optimizer()
        lr_scheduler = self.build_scheduler(optimizer)
        eval_metric = MeanAveragePrecision(class_metrics=self.class_metrics)

        for epoch in range(self.epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.epochs} ===")
            self.train_one_epoch(train_loader, optimizer, epoch, run)
            results = self.eval(val_loader, eval_metric, run, epoch)

            val_metrics = {
                "epoch": epoch + 1,
                "val/epoch/mAP": results["map"].item(),
                "val/epoch/mAP@50": results["map_50"].item(),
                "val/epoch/mAP@75": results["map_75"].item() if "map_75" in results else None,
                "val/epoch/mAR@100": results["mar_100"].item(),
                "val/epoch/mAP_small": results["map_small"].item() if "map_small" in results else None,
                "val/epoch/mAP_medium": results["map_medium"].item() if "map_medium" in results else None,
                "val/epoch/mAP_large": results["map_large"].item() if "map_large" in results else None,
            }
            val_metrics = {k: v for k, v in val_metrics.items() if v is not None}
            run.log(val_metrics)


        if self.class_metrics and "classes" in results and "map_per_class" in results:
            classes = results["classes"]
            map_pc = results["map_per_class"]
            mar_pc = results.get("mar_100_per_class", None)

            # Only proceed if these are 1D tensors with more than one class
            if classes.ndim == 1 and map_pc.ndim == 1 and classes.numel() == map_pc.numel():
                idx_to_class = (
                    self.val_dataset.idx_to_class
                    if hasattr(self.val_dataset, "idx_to_class")
                    else self.val_dataset.dataset.idx_to_class
                )

                mar_iter = mar_pc if (mar_pc is not None and mar_pc.ndim == 1) else None

                for i in range(classes.numel()):
                    cls_id = int(classes[i].item())
                    cls_map = float(map_pc[i].item())
                    cls_name = idx_to_class.get(cls_id, f"class_{cls_id}")

                    if cls_map >= 0:
                        run.summary[f"val/per_class/mAP/{cls_name}"] = cls_map

                    if mar_iter is not None:
                        cls_mar = float(mar_iter[i].item())
                        if cls_mar >= 0:
                            run.summary[f"val/per_class/mAR@100/{cls_name}"] = cls_mar
            else:
                print("[WARN] Skipping per-class logging: MAP metric returned scalars.")

            current_value = results[self.save_best_metric].item()
            if current_value > best_metric_value:
                best_metric_value = current_value
                checkpoint_path = RCNN_CHECKPOINTS_DIR / f"{runname}.pt"
                torch.save(self.model.state_dict(), checkpoint_path)

                artifact = wandb.Artifact(
                    name=f"{self.model_name}-best-model",
                    type="model",
                    metadata={
                        "model_name": self.model_name,
                        "dataset_name": self.dataset_name,
                        "best_metric_name": self.save_best_metric,
                        "best_metric_value": best_metric_value,
                        "epoch": epoch + 1,
                        "timestamp": timestamp,
                    },
                )
                artifact.add_file(str(checkpoint_path))
                run.log_artifact(artifact, aliases=["best"])

                run.summary["best_metric_name"] = self.save_best_metric
                run.summary["best_metric_value"] = best_metric_value
                run.summary["best_epoch"] = epoch + 1
                run.summary["best_model_path"] = str(checkpoint_path)

            if lr_scheduler is not None:
                lr_scheduler.step()

        wandb.finish()


if __name__ == "__main__":
    detector = RcnnDetector(
        model_name="fasterrcnn_mobilenet_v3_large_fpn",
        dataset_name="military",
        lr=0.001,
        epochs=12,
        batchsize=2,
        num_workers=1,
        img_size=640,
        augs="medium",
        trainable_backbone_layers=3,
        pretrained=True,
        weighted_sampling=True,
        optimizer_name="sgd",
        scheduler_name="none",
        class_metrics=True,
        subset_train_size=1,
        subset_val_size=1,
        use_amp=True
     
    )
    detector.train()