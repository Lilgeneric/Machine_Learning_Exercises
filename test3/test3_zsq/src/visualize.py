"""检测结果与 GT 可视化工具。"""
from __future__ import annotations

from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image


def _to_np_image(img) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().permute(1, 2, 0).numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        return arr
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    return np.asarray(img)


def draw_boxes(
    ax,
    boxes: torch.Tensor | np.ndarray,
    scores: Optional[Sequence[float]] = None,
    color: str = "lime",
    label_prefix: str = "pred",
    linewidth: float = 2.0,
):
    if hasattr(boxes, "cpu"):
        boxes = boxes.cpu().numpy()
    boxes = np.asarray(boxes).reshape(-1, 4)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=linewidth, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        if scores is not None:
            ax.text(
                x1, max(0, y1 - 4),
                f"{label_prefix} {scores[i]:.2f}",
                color="white", fontsize=8,
                bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", pad=1),
            )
        else:
            ax.text(
                x1, max(0, y1 - 4),
                label_prefix,
                color="white", fontsize=8,
                bbox=dict(facecolor=color, alpha=0.7, edgecolor="none", pad=1),
            )


def save_sample_with_gt(img, target: dict, save_path: str, title: str = ""):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(_to_np_image(img))
    draw_boxes(ax, target["boxes"], color="red", label_prefix="GT")
    ax.set_title(title or "Ground Truth")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_prediction_grid(
    img,
    gt_boxes,
    pred_boxes,
    pred_scores,
    thresholds: Sequence[float],
    save_path: str,
    title_prefix: str = "",
):
    """把不同阈值下的预测画成横向网格 + 左边一张 GT。"""
    n = len(thresholds) + 1
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]

    np_img = _to_np_image(img)

    axes[0].imshow(np_img)
    draw_boxes(axes[0], gt_boxes, color="red", label_prefix="GT")
    axes[0].set_title(f"{title_prefix}Ground Truth\n({len(gt_boxes)} boxes)")
    axes[0].axis("off")

    if hasattr(pred_scores, "cpu"):
        pred_scores = pred_scores.cpu().numpy()
    pred_scores = np.asarray(pred_scores)

    for ax, thr in zip(axes[1:], thresholds):
        ax.imshow(np_img)
        mask = pred_scores >= thr
        sel_boxes = pred_boxes[mask] if hasattr(pred_boxes, "shape") else [b for b, m in zip(pred_boxes, mask) if m]
        sel_scores = pred_scores[mask]
        draw_boxes(ax, sel_boxes, scores=sel_scores, color="lime", label_prefix="P")
        ax.set_title(f"threshold={thr:.2f}\n({int(mask.sum())} boxes)")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_loss_curve(
    epoch_losses: List[dict], save_path: str,
):
    """epoch_losses: 每个 epoch 的平均 loss 字典列表。"""
    keys = ["loss", "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    epochs = list(range(1, len(epoch_losses) + 1))
    for k in keys:
        vals = [e[k] for e in epoch_losses]
        ax.plot(epochs, vals, marker="o", label=k)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_iter_loss_curve(iter_losses: List[float], save_path: str):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(range(1, len(iter_losses) + 1), iter_losses, linewidth=0.8, color="steelblue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Loss")
    ax.set_title("Per-iteration Training Loss")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_map_curve(ap_history: List[float], save_path: str):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(range(1, len(ap_history) + 1), ap_history, marker="o", color="crimson")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AP@0.5 (val)")
    ax.set_title("Validation AP@0.5 per Epoch")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
