# Test3 — Penn-Fudan 行人检测（Faster R-CNN 迁移学习）

本项目实现机器学习作业 test3：在 Penn-Fudan Pedestrian 数据集上，基于 torchvision 的 COCO 预训练 Faster R-CNN（ResNet-50-FPN 骨干）进行迁移学习，训练一个两类（背景 + 行人）检测器。

## 项目结构

```
test3_zsq/
├── README.md                      本文件
├── Experiment_Report.md           实验报告（五个实验任务的回答与结果）
├── src/
│   ├── config.py                  全局超参、路径、线程数（CPU-only）
│   ├── dataset.py                 PennFudanDataset + transforms + collate_fn
│   ├── model.py                   build_model：加载 COCO 预训练 → 替换 box_predictor
│   ├── engine.py                  train_one_epoch / evaluate / box_iou
│   ├── visualize.py               GT/预测/loss/AP 曲线绘制
│   ├── train.py                   训练入口
│   └── inference.py               推理 + 阈值对比 + IoU 报告
└── outputs/
    ├── train.log                  训练日志（每 10 iter 打印 4 个 loss）
    ├── metrics.json               每 epoch 平均 loss、AP 历史
    ├── inference_report.json      推理阈值统计 + IoU 列表
    ├── checkpoints/{best,last}.pth
    ├── figures/                   loss_curve.png, map_curve.png, sample_with_gt.png ...
    └── predictions/               pred_thresh_grid_XX.png（不同阈值对比）
```

## 运行方式（全程 CPU）

> 本项目设计为不占用 GPU（`CUDA_VISIBLE_DEVICES=""`），不影响共享机器上其他 GPU 任务。

```bash
conda activate ml
cd test3/test3_zsq

# 冒烟测试（10 秒级）
python src/train.py --smoke-test --threads 4

# 完整训练（默认 20 epoch，CPU，~4-5 小时）
python src/train.py

# 自定义：更少 epoch / 更多线程
python src/train.py --epochs 10 --threads 16

# 推理 + 阈值对比（训练完成后）
python src/inference.py --num-samples 8
```

## 关键实现对照作业要求

### 一、数据层
- 从 `PedMasks/` 的实例掩码按像素 ID 提取 `[x_min, y_min, x_max, y_max]`（[src/dataset.py:39-47](src/dataset.py#L39-L47)）
- `__getitem__` 返回 `(image_tensor, {boxes, labels, image_id, area, iscrowd})`，符合 torchvision 约定
- `RandomHorizontalFlip` 同步翻转图像和 boxes（[src/dataset.py:78-88](src/dataset.py#L78-L88)）

### 二、模型层（迁移学习）
`build_model` 三步走（[src/model.py](src/model.py)）：
1. 加载 `fasterrcnn_resnet50_fpn(weights=COCO_V1)`（81 维预测头，80 类 + 背景）
2. 读取 `model.roi_heads.box_predictor.cls_score.in_features`
3. 用新的 `FastRCNNPredictor(in_features, num_classes=2)` 替换，保留骨干/RPN/FPN 的预训练权重

### 三、训练层
- SGD（lr=0.005, momentum=0.9, weight_decay=5e-4），MultiStepLR @ [12, 18]，γ=0.1
- 首个 epoch 前 500 iter 线性 warmup
- 每 10 iter 打印 4 个 loss：`loss_classifier` / `loss_box_reg` / `loss_objectness` / `loss_rpn_box_reg`
- 每 epoch 末在验证集上评估 mAP@0.5，保存 `best.pth`（按 val AP 选最优）

### 四、评价层
- `inference.py` 使用多个阈值（默认 0.3 / 0.5 / 0.7 / 0.9）
- 输出每张图在各阈值下保留的预测框数量 + IoU 值，生成对比图 `pred_thresh_grid_XX.png`
- 全局评估：VOC 11-point mAP（[src/engine.py:76-151](src/engine.py#L76-L151)）

### 五、结果输出
- `outputs/figures/sample_with_gt.png` — 原图 + GT 框
- `outputs/figures/loss_curve.png` — 每 epoch 4 个 loss 的曲线
- `outputs/figures/map_curve.png` — 验证集 AP@0.5 随 epoch 变化
- `outputs/predictions/pred_thresh_grid_XX.png` — 阈值对比检测图
- `outputs/inference_report.json` — IoU / 阈值计数详细数据

## 资源占用

- **GPU**：不使用（`CUDA_VISIBLE_DEVICES=""`）
- **CPU**：`torch.set_num_threads(16)`，`top` 中约 1600%（~50% 总容量）
- **RAM**：~3.5-4 GB
- 对共享机器上其他 GPU 训练/推理任务无影响

## 依赖

conda 环境 `ml`：`torch 2.5.1`, `torchvision 0.20.1`, `pycocotools 2.0.11`, `matplotlib`, `pillow`, `numpy`, `tqdm`
