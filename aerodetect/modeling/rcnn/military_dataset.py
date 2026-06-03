import os
import pandas as pd
import torch
from PIL import Image
from torchvision.io import read_image, ImageReadMode

from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from torchvision.tv_tensors import BoundingBoxes

from aerodetect.config import PROCESSED_DATA_DIR
from aerodetect.dataset import DatasetName
import yaml


class MilitaryDataset(Dataset):
    def __init__(
        self,
        split: str,
        img_size: int = 800,
        augment: bool = False,
    ):
        self.split = split
        self.img_size = img_size
        self.class_to_idx, self.idx_to_class = self.get_class_mapping()
        self.augment = augment

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
            "rcnn",
        )

        self.image_files = sorted(
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith(".jpg")
        )

        # --- Transforms (v2) ---
        self.transforms = self._build_transforms()

    # -------------------------
    # TRANSFORMS
    # -------------------------
    def _build_transforms(self):
        ops = []

        # Always resize per sample (your requirement)
        ops.append(T.ToImage())  # ensures tensor image format
        ops.append(T.Resize((self.img_size, self.img_size)))

        # Optional augmentations
        if self.augment:
            ops += [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomPhotometricDistort(),
            ]

        # Convert everything to tensor format (handled by v2)
        
        ops.append(T.ToDtype(torch.float32, scale=True))
    
        return T.Compose(ops)

    # -------------------------
    # LENGTH
    # -------------------------
    def __len__(self):
        return len(self.image_files)

    # -------------------------
    # GET ITEM
    # -------------------------
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        stem = os.path.splitext(image_file)[0]

        image_path = os.path.join(self.images_dir, image_file)
        label_path = os.path.join(self.labels_dir, f"{stem}.csv")

        # ---- Load image ----
        image = read_image(path=image_path, mode= ImageReadMode.RGB)

        C, H, W = image.shape
        # ---- Load annotations lazily (IMPORTANT FIX) ----
        df = pd.read_csv(label_path)

        boxes = torch.tensor(
            df[["xmin", "ymin", "xmax", "ymax"]].values,
            dtype=torch.float32,
        )

        labels = torch.tensor(
            [self.class_to_idx[c] for c in df["class"]],
            dtype=torch.int64,
        )

        # ---- Convert boxes to v2 format ----
        boxes = BoundingBoxes(
            boxes,
            format="XYXY",
            canvas_size=(H,W),  # (H, W)
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        # ---- Apply v2 transforms (image + boxes synced) ----
        image, target = self.transforms(image, target)

        return image, target

    # -------------------------
    # UTILITY
    # -------------------------
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
            "data.yaml"
        )

        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        names = cfg["names"]  # dict: {0: "Aircraft", ...}

        # Reserve 0 for background
        class_to_idx = {
            name: idx + 1
            for idx, name in names.items()
        }

        idx_to_class = {
            idx: name
            for name, idx in class_to_idx.items()
        }

        return class_to_idx, idx_to_class