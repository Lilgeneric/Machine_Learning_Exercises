"""检测任务专用的数据增广。

与分类任务不同, 几何变换会同时作用在图片与 boxes 上。
这里实现的增广全部基于 torch.Tensor, 与 torchvision Faster R-CNN 输入一致。

实现的增广:
- ToTensor            : PIL -> Tensor, 范围 [0,1], CHW
- RandomHorizontalFlip: 同步翻转 boxes
- ColorJitter         : 仅改变像素, 不修改 boxes
- RandomPhotoNoise    : 少量高斯噪声, 提升鲁棒性
"""
from __future__ import annotations

from typing import Callable, List

import numpy as np
import torch
from PIL import Image


class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class ToTensor:
    """PIL.Image -> float Tensor [C,H,W] 归一化到 [0,1]。"""

    def __call__(self, img, target):
        if isinstance(img, Image.Image):
            arr = np.array(img.convert("RGB"))
        else:
            arr = np.asarray(img)
        img_t = torch.as_tensor(arr, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img_t, target


class RandomHorizontalFlip:
    """水平翻转图片并同步更新 boxes 的 x 坐标。"""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: torch.Tensor, target: dict):
        if torch.rand(1).item() < self.p:
            img = torch.flip(img, dims=[-1])
            w = img.shape[-1]
            if target["boxes"].numel() > 0:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return img, target


class ColorJitter:
    """亮度/对比度/饱和度抖动 (仅像素级, 不影响 boxes)。"""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img: torch.Tensor, target: dict):
        if self.brightness > 0:
            factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.brightness
            img = (img * factor).clamp(0.0, 1.0)
        if self.contrast > 0:
            factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast
            mean = img.mean(dim=[-1, -2], keepdim=True)
            img = ((img - mean) * factor + mean).clamp(0.0, 1.0)
        if self.saturation > 0:
            factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.saturation
            gray = img.mean(dim=0, keepdim=True)
            img = (gray + (img - gray) * factor).clamp(0.0, 1.0)
        return img, target


class RandomGaussianNoise:
    """少量高斯噪声 (模拟传感器噪声), 概率触发。"""

    def __init__(self, p: float = 0.3, sigma: float = 0.01):
        self.p = p
        self.sigma = sigma

    def __call__(self, img: torch.Tensor, target: dict):
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(img) * self.sigma
            img = (img + noise).clamp(0.0, 1.0)
        return img, target


def build_transforms(train: bool) -> Compose:
    """训练集启用增广; 验证/测试只做 ToTensor。"""
    ts: List[Callable] = [ToTensor()]
    if train:
        ts += [
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            RandomGaussianNoise(p=0.25, sigma=0.008),
        ]
    return Compose(ts)
