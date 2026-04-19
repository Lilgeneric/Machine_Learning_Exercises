"""训练 / 评估循环。

train_one_epoch:
    - 对每个 iter 记录 total_loss 与 4 个分项 loss (classifier / box_reg / objectness / rpn_box_reg)
    - 第一个 epoch 自动启用线性 warmup, 避免预训练权重被初始大 LR 冲垮
    - 加入梯度裁剪, 防止偶发 NaN

evaluate:
    - 收集所有预测, 调用 metrics.compute_detection_metrics
    - 支持多 IoU 阈值 (0.5, 0.75, 以及 COCO 平均)
"""
from __future__ import annotations

import math
import time
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from metrics import box_iou, compute_detection_metrics


def warmup_lr_scheduler(optimizer, warmup_iters: int, warmup_factor: float):
    """线性 warmup 调度器 (LambdaLR). 第一个 epoch 内的每个 iter 都会 step 一次。"""
    def f(x):
        if x >= warmup_iters:
            return 1.0
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_one_epoch(
    model,
    optimizer,
    data_loader: DataLoader,
    device,
    epoch: int,
    print_every: int = 10,
    warmup: bool = True,
    warmup_factor: float = 1.0 / 1000,
    warmup_iters_cap: int = 500,
    grad_clip_max_norm: float = 0.0,
    logger=print,
) -> Dict[str, List[float]]:
    """训练一个 epoch, 返回每个 iter 的 loss 明细."""
    model.train()

    lr_scheduler = None
    if epoch == 0 and warmup:
        warmup_iters = min(warmup_iters_cap, len(data_loader) - 1)
        if warmup_iters > 0:
            lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    history = {
        "loss": [], "loss_classifier": [], "loss_box_reg": [],
        "loss_objectness": [], "loss_rpn_box_reg": [], "lr": [],
    }

    t0 = time.time()
    for it, (images, targets) in enumerate(data_loader):
        images = [img.to(device, non_blocking=True) for img in images]
        clean_targets = []
        for t in targets:
            clean_targets.append({
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in t.items() if k in ("boxes", "labels", "image_id", "area", "iscrowd")
            })

        loss_dict = model(images, clean_targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()
        if not math.isfinite(loss_value):
            logger(f"[epoch {epoch} it {it}] Loss 非有限值 ({loss_value}), 终止.")
            logger(loss_dict)
            raise RuntimeError("loss is not finite")

        optimizer.zero_grad()
        losses.backward()
        if grad_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        history["loss"].append(loss_value)
        history["loss_classifier"].append(loss_dict["loss_classifier"].item())
        history["loss_box_reg"].append(loss_dict["loss_box_reg"].item())
        history["loss_objectness"].append(loss_dict["loss_objectness"].item())
        history["loss_rpn_box_reg"].append(loss_dict["loss_rpn_box_reg"].item())
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if (it + 1) % print_every == 0 or it == 0:
            logger(
                f"[epoch {epoch}] iter {it+1:03d}/{len(data_loader)} "
                f"loss={loss_value:.4f} "
                f"cls={loss_dict['loss_classifier'].item():.4f} "
                f"box={loss_dict['loss_box_reg'].item():.4f} "
                f"obj={loss_dict['loss_objectness'].item():.4f} "
                f"rpn_box={loss_dict['loss_rpn_box_reg'].item():.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
            )

    logger(f"[epoch {epoch}] 单 epoch 训练耗时 {time.time() - t0:.1f}s")
    return history


@torch.no_grad()
def collect_predictions(
    model,
    data_loader: DataLoader,
    device,
):
    """把模型对 data_loader 每张图的预测收集起来.

    返回 list[dict], 每 dict 含:
        image_id    : int
        filename    : str
        pred_boxes  : [P, 4] tensor (cpu)
        pred_scores : [P]    tensor (cpu)
        pred_labels : [P]    tensor (cpu)
        gt_boxes    : [G, 4] tensor (cpu)
    """
    model.eval()
    all_results = []
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for output, target in zip(outputs, targets):
            all_results.append({
                "image_id": int(target["image_id"].item()),
                "filename": target.get("filename", ""),
                "pred_boxes": output["boxes"].detach().cpu(),
                "pred_scores": output["scores"].detach().cpu(),
                "pred_labels": output["labels"].detach().cpu(),
                "gt_boxes": target["boxes"].detach().cpu(),
            })
    return all_results


@torch.no_grad()
def evaluate(
    model,
    data_loader: DataLoader,
    device,
    iou_thresholds: List[float] = (0.5, 0.75),
    score_threshold: float = 0.05,
    logger=print,
) -> Dict[str, float]:
    """封装: 收集预测 -> 计算指标."""
    results = collect_predictions(model, data_loader, device)
    metrics = compute_detection_metrics(
        results, iou_thresholds=iou_thresholds, score_threshold=score_threshold
    )

    # per_image_best_iou 信息太长, 不打印
    pretty = {k: v for k, v in metrics.items()
              if k not in ("per_iou_ap", "per_image_best_iou")}
    logger("[eval] " + " ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in pretty.items()
    ))
    return metrics
