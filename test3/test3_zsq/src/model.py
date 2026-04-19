"""Faster R-CNN 模型构建器（COCO 预训练 → 微调为 2 类：背景 + 行人）。

迁移学习 3 步：
  1) 加载在 COCO (80 类) 上预训练的 fasterrcnn_resnet50_fpn
  2) 读取原检测头的输入通道数 in_features
  3) 替换 box_predictor 为新头部，输出 num_classes (=2)，其他权重保留
"""
from __future__ import annotations

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(num_classes: int = 2, pretrained: bool = True):
    weights = (
        torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        if pretrained else None
    )
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
