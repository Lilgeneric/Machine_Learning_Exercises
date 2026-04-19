# 迁移学习: 从 COCO 80 类到行人 2 类

## 为什么要迁移而不是从零训练

| 维度             | 从零 (Train from Scratch) | 迁移 (Fine-tune) |
| ---------------- | ------------------------- | ---------------- |
| 数据量需求        | 数万张 +                  | 几十到几百张即可 |
| 训练时长          | 数天 (8x GPU)             | 几分钟 (单卡)     |
| 底层特征          | 需要自己学 (边缘 / 纹理)   | COCO 已训练好    |
| Penn-Fudan 精度  | 收敛慢, 容易过拟合         | 2 min 达 AP 0.99 |

COCO 含 80 个类, 其中就有 "person", 因此 Penn-Fudan 这种"单类行人"几乎是 COCO 行人检测的子集, 迁移效果非常好.

## 关键冲突: 类别数

torchvision 的预训练权重在最后一层分类器输出 `num_classes=91` (80 类 + 背景 + 保留槽位).
我们只要 2 类 (1 行人 + 1 背景), 所以必须替换 `box_predictor`, 否则:

- **尺寸冲突**: `cls_score` 层的输出维度不匹配, forward 就炸.
- **语义错误**: 即使形状对上, 这个头学的是 "是不是车是不是狗", 不是 "是不是行人".

## 三步改造

```python
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision

# 1) 加载 COCO 权重
weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)

# 2) 读取原头的输入通道 (V2 是 1024)
in_features = model.roi_heads.box_predictor.cls_score.in_features

# 3) 换头: cls_score 和 bbox_pred 都会被重置
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
```

替换后:

- `cls_score`:  `[1024 → 2]`     (背景 / 行人)
- `bbox_pred`: `[1024 → 2*4=8]`  (per-class box 回归)

所有 backbone + FPN + RPN + 其他 head 权重都保留, 只有 box_predictor 是新初始化的.

## 可选: 冻结 backbone 的前几层

torchvision 的 `fasterrcnn_resnet50_fpn_v2(trainable_backbone_layers=k)` 控制 ResNet50 里多少个 stage 参与梯度更新 (k ∈ [0, 5]):

- `k = 0`: 完全冻结, 只训 head → 最节省资源, 表达力受限.
- `k = 3` (本实验默认): 最后 3 个 stage 可训, 在 170 张图上足够且不易过拟合.
- `k = 5`: 全部可训, 需要更多数据防止灾难性遗忘.

## 微调训练的超参建议

| 超参         | 建议值              | 为什么                                      |
| ----------- | ------------------ | ------------------------------------------ |
| optimizer   | SGD + momentum=0.9 | 与 Faster R-CNN 论文一致, 比 Adam 更稳       |
| lr          | 5e-3               | torchvision tutorial 推荐; 太大会毁预训练   |
| weight_decay| 5e-4               | 抑制过拟合                                  |
| warmup      | 线性, 第 1 epoch     | 避免大 lr 直接击穿预训练权重, 必做          |
| batch_size  | 2-4                | 检测模型显存密集, bs 太大易 OOM             |
| epochs      | 10-30              | 小数据集 10 epoch 已收敛, 继续训只有边际收益|

## 一句话总结

> **迁移学习 = "借来 COCO 已学会的底层感官, 换一个新的分类头, 再用少量目标数据微调."**

这也是为什么本实验能在 170 张图、2 分钟训练内达到 AP@0.5 = 99.34%.
