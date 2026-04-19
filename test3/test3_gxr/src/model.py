"""Faster R-CNN 模型构建器 (迁移学习, COCO -> 2 类)。

torchvision 提供了两个版本:
    v1: fasterrcnn_resnet50_fpn         (原论文风格, COCO mAP ~37.0)
    v2: fasterrcnn_resnet50_fpn_v2      (加强版 backbone, 更强 head, COCO mAP ~46.7)

我们默认使用 v2, 因为:
    - 同样的 fine-tune 代价, v2 起点更高, Penn-Fudan 这种小数据集收益明显
    - 已在 torchvision>=0.13 中稳定提供

迁移学习 3 步 (每一步在注释中解释"为什么"):
    1) 加载带 COCO 权重的检测模型 (80 类), 卷积特征提取已训练到位
    2) 读取原检测头的输入通道数 in_features
    3) 把 box_predictor 换成新的 FastRCNNPredictor(in_features, num_classes=2)
       这样模型就输出 "背景 / 行人" 两类, 其他权重保留
"""
from __future__ import annotations

from typing import Literal

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


ModelVariant = Literal["v1", "v2"]


def build_model(
    num_classes: int = 2,
    variant: ModelVariant = "v2",
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
) -> torch.nn.Module:
    """构造 Faster R-CNN 并替换分类头。

    参数:
        num_classes : 类别数 (含背景). Penn-Fudan 为 2.
        variant     : "v1" 或 "v2", v2 效果更好.
        pretrained  : 是否使用 COCO 预训练权重. 生产训练应 True, 结构验证可 False.
        trainable_backbone_layers:
            0-5. 控制 ResNet50 中有多少个 block 参与梯度更新.
            对于 170 张的小数据集, 默认 3 平衡了过拟合与表达能力.
    """
    if variant == "v2":
        weights = (
            torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
            if pretrained else None
        )
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=weights,
            weights_backbone=None,
            trainable_backbone_layers=trainable_backbone_layers,
        )
    elif variant == "v1":
        weights = (
            torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
            if pretrained else None
        )
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=None,
            trainable_backbone_layers=trainable_backbone_layers,
        )
    else:
        raise ValueError(f"未知 variant: {variant}")

    # ---- 关键: 替换最终分类头为 2 类 ----
    # 原 cls_score 形状 [81, in_features] (80 COCO + 1 背景),
    # 替换后 [num_classes, in_features]. bbox_pred 也会被替换为 [num_classes*4, ...].
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_M": round(total / 1e6, 2),
        "trainable_M": round(trainable / 1e6, 2),
    }
