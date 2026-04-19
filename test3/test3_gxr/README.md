# test3_gxr — Faster R-CNN Penn-Fudan Pedestrian Detection

机器学习实验三, 在 Penn-Fudan 行人数据集上微调 Faster R-CNN.

> 全部实验结果与分析见 [Experiment_Report.md](Experiment_Report.md).

## 核心结果

| 指标               | 值      |
| ----------------- | ------- |
| AP@0.5            | 0.9934  |
| AP@0.75           | 0.9687  |
| mAP@[0.5:0.95]    | 0.8850  |
| 最佳 F1 (thr=0.8) | 0.968   |
| 训练耗时          | ~2.1 min (RTX 4090) |

## 快速开始

```bash
# 1) 进入 ml 环境
conda activate ml

# 2) 从 test3_gxr/ 根目录训练
cd test3/test3_gxr
python src/train.py

# 3) 推理 + 阈值对比 + 生成全部图
python src/inference.py --num-samples 10
```

训练开始前会自动下载 torchvision 的 `fasterrcnn_resnet50_fpn_v2` COCO 预训练权重 (~167 MB, 之后缓存).

## 目录结构

```
test3_gxr/
├── Experiment_Report.md        # 完整实验报告 (指标 + 图 + 分析)
├── README.md                   # 本文件
├── docs/                       # 扩展文档
│   ├── architecture.md         # Faster R-CNN 架构速览
│   ├── transfer_learning.md    # 迁移学习三步
│   └── frcnn_vs_yolo.md        # 与 YOLO 的对比
├── src/
│   ├── config.py               # 超参数 / 路径 / 种子
│   ├── dataset.py              # Penn-Fudan 数据集 + mask→bbox
│   ├── transforms.py           # 检测专用增广 (几何 + 像素)
│   ├── model.py                # Faster R-CNN 构建器 (v1/v2) + 头替换
│   ├── engine.py               # train_one_epoch + evaluate
│   ├── metrics.py              # IoU, VOC11 AP, COCO 101 AP
│   ├── visualize.py            # 全部作图函数
│   ├── train.py                # 训练入口
│   └── inference.py            # 推理 + 阈值扫描 + 报告
└── outputs/
    ├── checkpoints/
    │   ├── best.pth            # val AP@0.5 最佳
    │   └── last.pth            # 最后 epoch
    ├── figures/                # GT 样例、loss 曲线、mAP 曲线、PR 曲线、IoU 直方图、阈值扫描
    ├── predictions/            # 逐张阈值对比 + GT vs Pred 对比
    ├── logs/                   # train.log / stdout
    ├── metrics.json            # 训练指标摘要
    └── inference_report.json   # 推理报告 (阈值扫描 + 每图统计)
```

## 与 test3_zsq 的差异 (gxr 的改进点)

| 项目                 | zsq                                | gxr                                              |
| -------------------- | ---------------------------------- | ----------------------------------------------- |
| 模型                 | fasterrcnn_resnet50_fpn (v1)       | fasterrcnn_resnet50_fpn_v2 (v2, 更强 head)       |
| 训练设备             | CPU (兼容)                         | CUDA (RTX 4090)                                 |
| 数据增广             | 水平翻转                          | 水平翻转 + ColorJitter + Gaussian Noise          |
| LR 调度             | MultiStepLR + warmup               | 同上 + 梯度裁剪 + 增大 warmup 容量              |
| 评估指标            | AP@0.5 (VOC11)                     | AP@0.5/0.75 + mAP@[0.5:0.95] (COCO 101 点)      |
| 可视化              | loss / map / threshold grid        | 同上 + PR 曲线 + IoU 直方图 + 阈值扫描 + LR 曲线 |
| 报告                | 单一 Report                        | 报告 + README + 架构/迁移学习/对比三篇 docs     |

## 主要命令参考

```bash
# 快速冒烟测试 (1 epoch, 验证流程, CPU 也能跑)
python src/train.py --smoke-test

# 用 v1 对比
python src/train.py --variant v1

# 推理指定阈值
python src/inference.py --thresholds 0.3 0.5 0.7 0.9

# 从 last.pth 推理
python src/inference.py --ckpt outputs/checkpoints/last.pth
```

## 依赖

- Python 3.10
- PyTorch 2.5.1 + CUDA 12.1
- torchvision 0.20.1
- matplotlib, numpy, Pillow

全部由 `ml` conda 环境提供.

## 参考

- [torchvision Faster R-CNN tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- Ren et al., *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*, NeurIPS 2015.
- Wang et al., *Object Detection Combining Recognition and Segmentation*, ACCV 2007 (Penn-Fudan).
