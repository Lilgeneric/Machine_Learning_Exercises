# Faster R-CNN 架构速览

> 目标: 把 forward pass 里 "发生了什么" 讲清楚, 方便对照 torchvision 的代码.

## 一句话描述

> Faster R-CNN = **Backbone (特征提取) + FPN (多尺度) + RPN (候选框) + RoIAlign (特征对齐) + Box Head (分类 + 回归)**.

## 前向流程

输入: 原图 `x ∈ [0,1]^{3×H×W}` (list).

```
1. GeneralizedRCNNTransform
     - 归一化 + Resize (短边 800, 长边 ≤ 1333)
     - 多张图 pad 到同一尺寸
2. Backbone: ResNet50
     - 5 个 stage 的 feature map {C2, C3, C4, C5}
3. FPN (Feature Pyramid Network)
     - 输出 {P2, P3, P4, P5, P6} (多尺度特征)
4. RPN (Region Proposal Network)
     - 每个 anchor 预测 (objectness, Δbox)
     - Top-K 后做 NMS, 得到 ~1000 个候选
5. RoIAlign
     - 把每个候选 RoI 投影到对应 FPN level, 双线性采样出 7×7 特征
6. Box Head
     - 两层 MLP (或 ConvFC, V2 版改进)
     - cls_score:    [num_classes]
     - bbox_pred:    [num_classes × 4]  (per-class 回归)
7. Post-process
     - softmax + per-class NMS
     - 输出 (boxes, labels, scores)
```

训练时损失有 4 项 (详见 [Experiment_Report.md](../Experiment_Report.md#32-四大-loss-的含义)):

- `loss_objectness`, `loss_rpn_box_reg` — 来自第 4 步 (RPN)
- `loss_classifier`, `loss_box_reg` — 来自第 6 步 (Box Head)

## V2 相比 V1 的改进

torchvision 0.13+ 的 `fasterrcnn_resnet50_fpn_v2` 引入了:

1. **Anchor 优化**: anchor sizes/aspect ratios 更贴合现代训练分布.
2. **Conv ResBlock Head**: Box head 从两层 MLP 替换为若干 Conv + FC, 感受野更大.
3. **更长训练时长的 COCO 权重**: torchvision 团队重训的高精度版本.

效果: COCO mAP 从 v1 的 37.0 提升到 v2 的 46.7.

## 与 torchvision 对应的代码位置

| 模块          | torchvision 文件                                      |
| ------------- | ---------------------------------------------------- |
| GeneralizedRCNNTransform | `torchvision/models/detection/transform.py` |
| Backbone + FPN | `torchvision/models/detection/backbone_utils.py`    |
| RPN           | `torchvision/models/detection/rpn.py`                |
| RoI head      | `torchvision/models/detection/roi_heads.py`          |
| FastRCNNPredictor | `torchvision/models/detection/faster_rcnn.py`    |
