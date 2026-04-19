"""全局配置 (gxr 版)。

设计理念:
- 训练超参、路径、评估阈值集中管理, 避免散落到各脚本.
- 默认使用 CUDA (检测到 GPU 时), 失败回落到 CPU.
- 为了复现性, 所有随机种子都在一处管理.
"""
from __future__ import annotations

import os

import torch as _torch


# ---------- 路径 ----------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "PennFudanPed"))

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")


# ---------- 设备 ----------
DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"


# ---------- 模型 ----------
# 2 = 背景 + 行人
NUM_CLASSES = 2
# "v1" = fasterrcnn_resnet50_fpn (COCO_V1),  "v2" = fasterrcnn_resnet50_fpn_v2 (COCO_V1, 更强)
MODEL_VARIANT = "v2"
TRAINABLE_BACKBONE_LAYERS = 3


# ---------- 数据 ----------
VAL_RATIO = 0.2
SEED = 42
# GPU 下 batch=4 很稳, CPU 回落 2
BATCH_SIZE = 4 if DEVICE == "cuda" else 2
NUM_WORKERS = 4 if DEVICE == "cuda" else 2


# ---------- 优化器 / LR ----------
# 使用 SGD + momentum, 是 Faster R-CNN 论文与 torchvision 官方示例的默认选择
NUM_EPOCHS = 15
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# LR 调度: 使用分段下降 (MultiStepLR), 在 2/3 与 5/6 处下降
LR_SCHEDULER = "multistep"   # "multistep" | "cosine"
LR_STEP = [10, 13]
LR_GAMMA = 0.1

# Warmup (仅第 1 epoch 使用, 线性 warmup 避免预训练权重被大 LR 破坏)
WARMUP_FACTOR = 1.0 / 1000
WARMUP_ITERS_CAP = 500


# ---------- 梯度裁剪 ----------
GRAD_CLIP_MAX_NORM = 5.0  # 防止偶发 NaN


# ---------- 评估 ----------
EVAL_IOU_THRESHOLDS = [0.5, 0.75]
EVAL_SCORE_THRESHOLD_FOR_VAL = 0.05  # 评估时保留的最低分数 (保持召回率以供 PR 曲线)
VIS_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
THRESHOLD_SWEEP = [round(x, 2) for x in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]


# ---------- CPU 线程 (在多卡机上占用太多线程会拖慢其他任务, 保持合理) ----------
TORCH_THREADS = 8
TORCH_INTEROP_THREADS = 2


def make_dirs() -> None:
    for d in (OUTPUT_DIR, CKPT_DIR, FIG_DIR, PRED_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)


def summary() -> str:
    return (
        f"device={DEVICE} variant={MODEL_VARIANT} epochs={NUM_EPOCHS} "
        f"bs={BATCH_SIZE} lr={LEARNING_RATE} sched={LR_SCHEDULER} "
        f"step={LR_STEP} gamma={LR_GAMMA} workers={NUM_WORKERS} seed={SEED}"
    )
