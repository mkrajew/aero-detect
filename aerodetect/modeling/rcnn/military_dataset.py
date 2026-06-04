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



class MilitaryDataset(Dataset):
    def __init__(
        self,
        split: str,
        img_size: int = 800,
        augment: bool = False,
    ):
        self.split = split
        self.img_size = img_size
        self.augment = augment

        self.class_to_idx, self.idx_to_class = self.get_class_mapping()

        self.images_dir = os.path.join(
            str(PROCESSED_DATA_DIR),
            str(DatasetName.military.name),
            "images",
            split,
        )

        self.labels_dir = os.path.join(
            str(PROCESSED_DATA_DIR),
            str(DatasetName.military.name),
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
        ]

        if self.augment:
            ops.extend(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomPhotometricDistort(),
                ]
            )

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
            str(DatasetName.military.name),
            "data.yaml",
        )

        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        return cfg["nc"] + 1

    @staticmethod
    def get_class_mapping():
        path = os.path.join(
            str(PROCESSED_DATA_DIR),
            str(DatasetName.military.name),
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
        Build filename -> sampling weight for a split, using the raw CSV and
        the same class mapping as the processed YOLO dataset.

        Returns:
            weight_map: dict[str, float]
            class_counts: dict[str, int]
        """
        labels_csv = RAW_DATA_DIR / "military" / "labels_with_split.csv"

        if not labels_csv.is_file():
            raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

        df = pd.read_csv(labels_csv)
        df = df[df["split"] == self.split].copy()

        if df.empty:
            raise ValueError(f"No rows found for split='{self.split}'")

        class_to_idx, idx_to_class = MilitaryDataset.get_class_mapping()

        # Optional sanity check: all CSV classes must exist in YAML mapping
        csv_classes = set(df["class"].unique())
        yaml_classes = set(class_to_idx.keys())
        missing = csv_classes - yaml_classes
        if missing:
            raise ValueError(f"Classes in CSV missing from data.yaml mapping: {sorted(missing)}")

        class_counts = Counter(df["class"].tolist())

        # inverse-frequency weights by class name
        class_weights = {
            cls_name: 1.0 / count
            for cls_name, count in class_counts.items()
            if count > 0
        }

        weight_map = {}
        for filename, rows in df.groupby("filename", sort=False):
            labels = rows["class"].tolist()
            label_weights = [class_weights[label] for label in labels]

            if mode == "sum":
                weight = sum(label_weights)
            else:
                weight = max(label_weights)

            weight_map[filename] = float(weight)

        # optional normalization
        if weight_map:
            mean_weight = sum(weight_map.values()) / len(weight_map)
            weight_map = {k: v / mean_weight for k, v in weight_map.items()}

        dataset_weights = []
        for filename in self.image_files:
            dataset_weights.append(weight_map[os.path.splitext(filename)[0]])
        

        return dataset_weights