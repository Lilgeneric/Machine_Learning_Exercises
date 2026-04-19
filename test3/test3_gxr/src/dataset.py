"""Penn-Fudan 行人检测数据集。

官方 readme 说明掩码编码:
    像素值 0  = 背景
    像素值 i  = 第 i 个行人实例 (i >= 1)

因此从掩码生成 bbox 的关键步骤:
    1. np.unique(mask) 获取所有实例 id (剔除 0)
    2. 对每个实例 id, 取 mask == id 的像素坐标
    3. 用 (xs.min, ys.min, xs.max, ys.max) 构造 [x_min, y_min, x_max, y_max]

torchvision Faster R-CNN 约定的 target 字段:
    boxes    : FloatTensor [N, 4]  绝对像素坐标 (x1,y1,x2,y2)
    labels   : Int64Tensor [N]     类别 id (>=1 表示前景)
    image_id : Int64Tensor [1]
    area     : FloatTensor [N]
    iscrowd  : Int64Tensor [N]     本数据集全部为 0 (不是群体)

__getitem__ 返回 (image_tensor, target_dict), DataLoader 的 collate_fn 需要
tuple(zip(*batch)), 所以我们在 dataset.py 同文件提供 collate_fn。
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PennFudanDataset(Dataset):
    """Penn-Fudan Pedestrian Database。

    参数:
        root        : 数据集根目录, 内部需存在 PNGImages/ 与 PedMasks/
        transforms  : 接收 (img, target) 并返回 (img, target) 的增广管线
        indices     : 可选索引子集, 便于在同一个 root 下做 train/val 划分
    """

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        indices: Optional[List[int]] = None,
    ):
        self.root = root
        self.transforms = transforms

        all_imgs = sorted(os.listdir(os.path.join(root, "PNGImages")))
        all_masks = sorted(os.listdir(os.path.join(root, "PedMasks")))
        assert len(all_imgs) == len(all_masks), \
            f"图片与掩码数量不一致: imgs={len(all_imgs)} masks={len(all_masks)}"

        if indices is None:
            indices = list(range(len(all_imgs)))

        self.imgs = [all_imgs[i] for i in indices]
        self.masks = [all_masks[i] for i in indices]
        self._orig_indices = list(indices)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path))  # uint8, H x W

        # 从实例掩码提取每个行人的 bbox
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]  # 去掉背景

        boxes: List[List[float]] = []
        for oid in obj_ids:
            ys, xs = np.where(mask == oid)
            if xs.size == 0:
                continue
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            # torchvision 要求 x2 > x1, y2 > y1, 否则会抛 IndexError
            if x_max > x_min and y_max > y_min:
                boxes.append([float(x_min), float(y_min), float(x_max), float(y_max)])

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.ones((boxes_t.shape[0],), dtype=torch.int64)  # 1 = pedestrian
        areas = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        iscrowd = torch.zeros((boxes_t.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels,
            "image_id": torch.tensor([self._orig_indices[idx]], dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd,
            "orig_size": torch.tensor([img.height, img.width], dtype=torch.int64),
            "filename": self.imgs[idx],
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img = torch.as_tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0

        # 过滤 target 中不能堆叠的键 (如 filename / orig_size 不参与前向)
        return img, target


def collate_fn(batch):
    """Faster R-CNN 的输入是 list[Tensor] 与 list[dict], 不能做 torch.stack。"""
    return tuple(zip(*batch))


def train_val_split(
    total: int, val_ratio: float, seed: int
) -> Tuple[List[int], List[int]]:
    """按固定 seed 做 index 级的划分, 避免 train/val 出现同张图。"""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total, generator=g).tolist()
    n_val = max(1, int(total * val_ratio))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return sorted(train_idx), sorted(val_idx)
