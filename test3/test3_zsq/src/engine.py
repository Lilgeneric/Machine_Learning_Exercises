"""训练/评估循环、IoU 与简单 mAP@0.5 计算。

所有函数都是 pure function，方便在 train.py / evaluate.py 中复用。
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader


def warmup_lr_scheduler(optimizer, warmup_iters: int, warmup_factor: float):
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
    logger=print,
) -> Dict[str, List[float]]:
    model.train()

    lr_scheduler = None
    if epoch == 0 and warmup:
        warmup_iters = min(500, len(data_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, 1.0 / 1000)

    history = {
        "loss": [], "loss_classifier": [], "loss_box_reg": [],
        "loss_objectness": [], "loss_rpn_box_reg": [], "lr": [],
    }

    for it, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()
        if not math.isfinite(loss_value):
            logger(f"[epoch {epoch} it {it}] Loss 非有限值 ({loss_value})，终止。")
            logger(loss_dict)
            raise RuntimeError("loss is not finite")

        optimizer.zero_grad()
        losses.backward()
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
    return history


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """标准 IoU 矩阵，输入 [N,4] 与 [M,4]，返回 [N,M]。"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-9)


@torch.no_grad()
def evaluate(
    model,
    data_loader: DataLoader,
    device,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    logger=print,
) -> Dict[str, float]:
    """单类 VOC-style mAP@iou_threshold（11-point interpolation）。"""
    model.eval()

    all_scores: List[float] = []
    all_tp: List[int] = []
    all_fp: List[int] = []
    total_gt = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            gt_boxes = target["boxes"].to(device)
            total_gt += gt_boxes.shape[0]

            pred_boxes = output["boxes"]
            pred_scores = output["scores"]

            keep = pred_scores >= score_threshold
            pred_boxes = pred_boxes[keep]
            pred_scores = pred_scores[keep]

            if pred_boxes.shape[0] == 0:
                continue

            order = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[order]
            pred_scores = pred_scores[order]

            matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=device)

            if gt_boxes.shape[0] == 0:
                for s in pred_scores:
                    all_scores.append(s.item())
                    all_tp.append(0)
                    all_fp.append(1)
                continue

            ious = box_iou(pred_boxes, gt_boxes)

            for i in range(pred_boxes.shape[0]):
                best_iou, best_j = ious[i].max(0)
                if best_iou.item() >= iou_threshold and not matched[best_j]:
                    matched[best_j] = True
                    all_tp.append(1)
                    all_fp.append(0)
                else:
                    all_tp.append(0)
                    all_fp.append(1)
                all_scores.append(pred_scores[i].item())

    if total_gt == 0 or len(all_scores) == 0:
        logger("[evaluate] 无 GT 或无预测，返回 0。")
        return {"AP": 0.0, "precision": 0.0, "recall": 0.0, "total_gt": total_gt}

    scores_t = torch.tensor(all_scores)
    tp_t = torch.tensor(all_tp, dtype=torch.float32)
    fp_t = torch.tensor(all_fp, dtype=torch.float32)
    order = torch.argsort(scores_t, descending=True)
    tp_t = tp_t[order]
    fp_t = fp_t[order]

    tp_cum = torch.cumsum(tp_t, dim=0)
    fp_cum = torch.cumsum(fp_t, dim=0)
    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / (tp_cum + fp_cum).clamp(min=1e-9)

    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max().item() / 11.0

    final_prec = precisions[-1].item()
    final_rec = recalls[-1].item()
    logger(
        f"[evaluate] AP@{iou_threshold}={ap:.4f} "
        f"precision={final_prec:.4f} recall={final_rec:.4f} "
        f"pred={len(all_scores)} gt={total_gt}"
    )
    return {
        "AP": ap,
        "precision": final_prec,
        "recall": final_rec,
        "total_gt": total_gt,
        "total_pred": len(all_scores),
    }
