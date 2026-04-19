"""全局训练配置（CPU-only，避免影响共享的 GPU）。"""
from __future__ import annotations

import os


DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "PennFudanPed"))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")

import torch as _torch
DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"

NUM_CLASSES = 2
BATCH_SIZE = 4 if DEVICE == "cuda" else 2
NUM_WORKERS = 4 if DEVICE == "cuda" else 2

NUM_EPOCHS = 20
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

LR_STEP = [12, 18]
LR_GAMMA = 0.1

SEED = 42
VAL_RATIO = 0.2
TORCH_THREADS = 16
TORCH_INTEROP_THREADS = 2

EVAL_IOU_THRESHOLD = 0.5
EVAL_SCORE_THRESHOLD = 0.05
VIS_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]


def make_dirs():
    for d in (OUTPUT_DIR, CKPT_DIR, FIG_DIR, PRED_DIR):
        os.makedirs(d, exist_ok=True)
