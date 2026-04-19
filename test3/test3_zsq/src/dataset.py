"""PennFudan 行人检测数据集。

掩码语义：像素值 0 = 背景；像素值 i (i>0) = 第 i 个行人实例。
__getitem__ 返回 (image_tensor, target_dict)，符合 torchvision Faster R-CNN 输入约定。
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PennFudanDataset(Dataset):
    def __init__(self, root: str, transforms: Optional[Callable] = None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(os.listdir(os.path.join(root, "PNGImages")))
        self.masks = sorted(os.listdir(os.path.join(root, "PedMasks")))
        assert len(self.imgs) == len(self.masks), "图片和掩码数量不一致"

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path))

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]
        num_objs = len(obj_ids)

        boxes = []
        for oid in obj_ids:
            ys, xs = np.where(mask == oid)
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            if x_max > x_min and y_max > y_min:
                boxes.append([float(x_min), float(y_min), float(x_max), float(y_max)])

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img = torch.as_tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0

        return img, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class ToTensor:
    def __call__(self, img, target):
        img = torch.as_tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img, target


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img, target):
        if torch.rand(1).item() < self.p:
            img = torch.flip(img, dims=[-1])
            w = img.shape[-1]
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes
        return img, target


def build_transforms(train: bool):
    ts = [ToTensor()]
    if train:
        ts.append(RandomHorizontalFlip(0.5))
    return Compose(ts)


def collate_fn(batch):
    return tuple(zip(*batch))
