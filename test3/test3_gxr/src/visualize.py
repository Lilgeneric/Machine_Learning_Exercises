"""可视化工具集: 样例图、loss 曲线、mAP 曲线、阈值对比、PR 曲线、IoU 直方图。

所有函数都接收 save_path, 直接写出 PNG; 调用方只管组合数据即可。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # 无 X11 环境也能保存图片
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image


# ---------------- 公共辅助 ----------------
def _to_np_image(img) -> np.ndarray:
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().permute(1, 2, 0).numpy()
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    return np.asarray(img)


def draw_boxes(
    ax,
    boxes,
    scores: Optional[Sequence[float]] = None,
    color: str = "lime",
    label_prefix: str = "pred",
    linewidth: float = 2.0,
    fontsize: int = 9,
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
                color="white", fontsize=fontsize,
                bbox=dict(facecolor=color, alpha=0.75, edgecolor="none", pad=1),
            )
        else:
            ax.text(
                x1, max(0, y1 - 4), label_prefix,
                color="white", fontsize=fontsize,
                bbox=dict(facecolor=color, alpha=0.75, edgecolor="none", pad=1),
            )


# ---------------- 数据层: GT 可视化 ----------------
def save_sample_with_gt(img, target: dict, save_path: str, title: str = ""):
    """单张图 + GT 矩形框. 用于作业 "生成四张数据集中的带 box 标注原图"."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(_to_np_image(img))
    draw_boxes(ax, target["boxes"], color="red", label_prefix="GT")
    ax.set_title(title or f"Ground Truth ({target['boxes'].shape[0]} ped)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_gt_grid(samples: List[tuple], save_path: str, ncols: int = 2):
    """把多张 (img, target) 打包成一个网格, 每张都画 GT 框."""
    n = len(samples)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (img, target, title) in zip(axes, samples):
        ax.imshow(_to_np_image(img))
        draw_boxes(ax, target["boxes"], color="red", label_prefix="GT")
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for ax in axes[len(samples):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------- 训练层: loss / mAP 曲线 ----------------
def save_loss_curve(epoch_losses: List[dict], save_path: str):
    """每个 epoch 的 4 项 loss + 总 loss 曲线."""
    keys = ["loss", "loss_classifier", "loss_box_reg",
            "loss_objectness", "loss_rpn_box_reg"]
    colors = ["black", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    epochs = list(range(1, len(epoch_losses) + 1))
    for k, c in zip(keys, colors):
        vals = [e[k] for e in epoch_losses]
        ax.plot(epochs, vals, marker="o", label=k, color=c,
                linewidth=2 if k == "loss" else 1.3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve (epoch average)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_iter_loss_curve(iter_losses: List[float], save_path: str, smooth_window: int = 10):
    """每个 iter 的 loss + 滑动平均."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    x = np.arange(1, len(iter_losses) + 1)
    ax.plot(x, iter_losses, color="lightsteelblue", linewidth=0.7, label="per-iter")
    if len(iter_losses) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(iter_losses, kernel, mode="valid")
        ax.plot(
            np.arange(smooth_window, len(iter_losses) + 1), smoothed,
            color="navy", linewidth=1.6, label=f"SMA-{smooth_window}",
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Loss")
    ax.set_title("Per-iteration Training Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_map_curve(ap_history: Dict[str, List[float]], save_path: str):
    """多条 AP 曲线: AP@0.5, AP@0.75, mAP@[0.5:0.95]."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    colors = {"AP@0.5": "crimson", "AP@0.75": "darkorange",
              "mAP@[0.5:0.95]": "steelblue"}
    for k, vals in ap_history.items():
        ax.plot(range(1, len(vals) + 1), vals, marker="o",
                label=k, color=colors.get(k))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AP (val)")
    ax.set_title("Validation AP per Epoch")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_lr_curve(lrs: List[float], save_path: str):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(range(1, len(lrs) + 1), lrs, color="teal")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Learning Rate (log)")
    ax.set_title("LR Schedule (warmup + MultiStepLR)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------- 推理层: 阈值对比 ----------------
def save_prediction_grid(
    img, gt_boxes, pred_boxes, pred_scores,
    thresholds: Sequence[float],
    save_path: str,
    title_prefix: str = "",
):
    """左 1 张 GT + 右 N 张不同置信度阈值下的预测."""
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
    if hasattr(pred_boxes, "cpu"):
        pred_boxes_np = pred_boxes.cpu().numpy()
    else:
        pred_boxes_np = np.asarray(pred_boxes)

    for ax, thr in zip(axes[1:], thresholds):
        ax.imshow(np_img)
        mask = pred_scores >= thr
        sel_boxes = pred_boxes_np[mask]
        sel_scores = pred_scores[mask]
        draw_boxes(ax, sel_boxes, scores=sel_scores,
                   color="lime", label_prefix="P")
        ax.set_title(f"threshold={thr:.2f}\n({int(mask.sum())} boxes)")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_side_by_side(
    img, gt_boxes, pred_boxes, pred_scores, threshold: float, save_path: str,
    title: str = "",
):
    """单阈值下的 GT vs 预测 对比图 (2x1)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    np_img = _to_np_image(img)

    axes[0].imshow(np_img)
    draw_boxes(axes[0], gt_boxes, color="red", label_prefix="GT")
    axes[0].set_title(f"Ground Truth ({len(gt_boxes)})")
    axes[0].axis("off")

    if hasattr(pred_scores, "cpu"):
        pred_scores = pred_scores.cpu().numpy()
    if hasattr(pred_boxes, "cpu"):
        pred_boxes = pred_boxes.cpu().numpy()
    mask = pred_scores >= threshold

    axes[1].imshow(np_img)
    draw_boxes(axes[1], pred_boxes[mask], scores=pred_scores[mask],
               color="lime", label_prefix="P")
    axes[1].set_title(f"Predictions @ score>={threshold:.2f} ({int(mask.sum())})")
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------- 评价层: PR 曲线 / IoU 直方图 / 阈值扫描 ----------------
def save_pr_curve(pr_data: Dict[str, tuple], save_path: str):
    """pr_data: { "AP@0.5": (precisions, recalls, ap), "AP@0.75": (...)}."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    for label, (p, r, ap) in pr_data.items():
        if len(p) == 0:
            continue
        ax.plot(r.numpy(), p.numpy(), linewidth=2, label=f"{label} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_iou_histogram(ious: List[float], save_path: str, iou_threshold: float = 0.5):
    """最佳 IoU 直方图 (每个 GT 对应的最佳预测 IoU)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    if len(ious) == 0:
        ax.text(0.5, 0.5, "No IoUs", ha="center", va="center")
    else:
        arr = np.array(ious)
        ax.hist(arr, bins=np.linspace(0, 1, 21), color="steelblue",
                edgecolor="white", alpha=0.85)
        ax.axvline(iou_threshold, color="crimson", linestyle="--", linewidth=1.6,
                   label=f"IoU threshold = {iou_threshold}")
        ax.axvline(arr.mean(), color="darkgreen", linestyle=":", linewidth=1.6,
                   label=f"mean = {arr.mean():.3f}")
        ax.legend()
    ax.set_xlabel("IoU (best match per GT)")
    ax.set_ylabel("Count")
    ax.set_title(f"IoU Distribution of Validation Predictions (n={len(ious)})")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_threshold_sweep(
    thresholds: List[float],
    precisions: List[float],
    recalls: List[float],
    f1s: List[float],
    counts: List[int],
    save_path: str,
):
    """阈值 vs (P, R, F1, pred 数量) 的扫描曲线."""
    fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))
    ax1.plot(thresholds, precisions, "o-", color="crimson", label="Precision")
    ax1.plot(thresholds, recalls, "s-", color="steelblue", label="Recall")
    ax1.plot(thresholds, f1s, "^-", color="darkgreen", label="F1")
    ax1.set_xlabel("Score threshold")
    ax1.set_ylabel("Precision / Recall / F1")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower left")

    ax2 = ax1.twinx()
    ax2.plot(thresholds, counts, "d--", color="gray", alpha=0.8,
             label="# predictions")
    ax2.set_ylabel("# predictions kept")
    ax2.legend(loc="upper right")

    ax1.set_title("Threshold Sweep on Validation Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
