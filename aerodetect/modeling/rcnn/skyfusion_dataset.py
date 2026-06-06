import os
import yaml

import torch
from torch.utils.data import Dataset

from torchvision.io import read_image, ImageReadMode
import torchvision.transforms.v2 as T
from torchvision.tv_tensors import BoundingBoxes

from aerodetect.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from aerodetect.dataset import DatasetName
from collections import Counter
from pathlib import Path
import pandas as pd



class SkyFusionDataset(Dataset):
    def __init__(
        self,
        split: str,
        img_size: int = 640,
        augment: str = "",
    ):
        self.split = split
        self.img_size = img_size
        self.augment = augment

        self.class_to_idx, self.idx_to_class = self.get_class_mapping()

        self.images_dir = os.path.join(
            str(PROCESSED_DATA_DIR),
            str(DatasetName.skyfusion.name),
            "images",
            split,
        )

        self.labels_dir = os.path.join(
            str(PROCESSED_DATA_DIR),
            str(DatasetName.skyfusion.name),
            "labels",
            split,
        )

        self.image_files = sorted(
            f
            for f in os.listdir(self.images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

       

        self.transforms = self._build_transforms()
    def _build_transforms(self):
        ops = [
            T.ToImage(),
            T.Resize((self.img_size, self.img_size)),
            T.ClampBoundingBoxes(),
        ]

        if self.augment == "light":
            ops.extend([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            ])

        elif self.augment == "medium":
            ops.extend([
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
                T.RandomPhotometricDistort(p=0.5),
            ])

        ops.append(T.ToDtype(torch.float32, scale=True))
        return T.Compose(ops)

    def __len__(self):
        return len(self.image_files)

    @staticmethod
    def _load_yolo_annotations(
        label_path: str,
        image_width: int,
        image_height: int,
    ):
        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    cls, xc, yc, w, h = map(
                        float,
                        line.split(),
                    )

                    cls = int(cls)

                    xmin = (xc - w / 2) * image_width
                    ymin = (yc - h / 2) * image_height
                    xmax = (xc + w / 2) * image_width
                    ymax = (yc + h / 2) * image_height

                    boxes.append(
                        [xmin, ymin, xmax, ymax]
                    )

                    # TorchVision reserves 0 for background
                    labels.append(cls + 1)

        boxes = torch.tensor(
            boxes,
            dtype=torch.float32,
        ).reshape(-1, 4)

        labels = torch.tensor(
            labels,
            dtype=torch.int64,
        )

        return boxes, labels

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        stem = os.path.splitext(image_file)[0]

        image_path = os.path.join(
            self.images_dir,
            image_file,
        )

        label_path = os.path.join(
            self.labels_dir,
            f"{stem}.txt",
        )

        image = read_image(
            image_path,
            mode=ImageReadMode.RGB,
        )

        _, H, W = image.shape

        boxes, labels = self._load_yolo_annotations(
            label_path,
            image_width=W,
            image_height=H,
        )

        boxes = BoundingBoxes(
            boxes,
            format="XYXY",
            canvas_size=(H, W),
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        image, target = self.transforms(
            image,
            target,
        )

        return image, target

    @staticmethod
    def get_class_number():
        path = os.path.join(
            str(PROCESSED_DATA_DIR),
            str(DatasetName.skyfusion.name),
            "data.yaml",
        )

        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        return cfg["nc"] + 1
           
    @staticmethod
    def get_class_mapping():
        path = os.path.join(
            str(PROCESSED_DATA_DIR),
            str(DatasetName.skyfusion.name),
            "data.yaml",
        )

        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        names = cfg["names"]

        if isinstance(names, list):
            names = {
                i: name
                for i, name in enumerate(names)
            }

        class_to_idx = {
            name: idx + 1
            for idx, name in names.items()
        }

        idx_to_class = {
            idx: name
            for name, idx in class_to_idx.items()
        }

        return class_to_idx, idx_to_class
    
    def build_sampling_weight_map(self, mode="max"):
        """
        Build per-image sampling weights directly from YOLO label files in this split.

        Args:
            mode:
                - "max": image weight = max inverse-frequency of labels in that image
                - "sum": image weight = sum inverse-frequency of labels in that image

        Returns:
            dataset_weights: list[float], aligned with self.image_files
        """
        if mode not in {"max", "sum"}:
            raise ValueError(f"Unsupported mode='{mode}'. Use 'max' or 'sum'.")

        image_label_lists = {}
        class_counts = Counter()

        for image_file in self.image_files:
            stem = os.path.splitext(image_file)[0]
            label_path = os.path.join(self.labels_dir, f"{stem}.txt")

            labels_in_image = []

            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) < 5:
                            continue

                        cls = int(float(parts[0])) + 1  # shift to TorchVision label space
                        labels_in_image.append(cls)
                        class_counts[cls] += 1

            image_label_lists[image_file] = labels_in_image

        if not class_counts:
            raise ValueError(f"No annotations found for split='{self.split}' in {self.labels_dir}")

        class_weights = {
            cls: 1.0 / count
            for cls, count in class_counts.items()
            if count > 0
        }

        weight_map = {}
        for image_file, labels in image_label_lists.items():
            if len(labels) == 0:
                weight = 1.0
            else:
                label_weights = [class_weights[label] for label in labels]

                if mode == "sum":
                    weight = sum(label_weights)
                else:
                    weight = max(label_weights)

            weight_map[image_file] = float(weight)

        mean_weight = sum(weight_map.values()) / len(weight_map)
        if mean_weight > 0:
            weight_map = {k: v / mean_weight for k, v in weight_map.items()}

        dataset_weights = [weight_map[image_file] for image_file in self.image_files]
        return dataset_weights