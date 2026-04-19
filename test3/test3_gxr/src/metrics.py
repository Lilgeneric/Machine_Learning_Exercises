"""检测评估指标: IoU, VOC-11 AP, 简化 COCO-style mAP。

所有函数接收 tensor 作为输入, 返回 tensor 或纯 python 数值, 方便 pickle。

本模块不依赖 pycocotools, 全部用 torch 原生算子实现, 便于理解与调试。
"""
from __future__ import annotations

from typing import Dict, List

import torch


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """计算两组 boxes 间的 IoU 矩阵.

    输入:
        boxes1: [N, 4]  (x1,y1,x2,y2)
        boxes2: [M, 4]
    输出:
        iou   : [N, M]  每行 i 表示 boxes1[i] 与所有 boxes2 的 IoU.
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-9)


def _ap_voc11(precisions: torch.Tensor, recalls: torch.Tensor) -> float:
    """VOC 2007 风格的 11 点插值 AP.

    对 recall 在 {0, 0.1, ..., 1.0} 上取值, 每点取该 recall 以上的最大 precision,
    再求平均.
    """
    ap = 0.0
    for t in torch.linspace(0.0, 1.0, 11):
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max().item() / 11.0
    return ap


def _ap_coco(precisions: torch.Tensor, recalls: torch.Tensor) -> float:
    """COCO 风格的 101 点插值 AP (更平滑的估计).

    对 recall 在 {0, 0.01, ..., 1.0} 上插值, 每点取该 recall 以上的最大 precision.
    """
    if precisions.numel() == 0:
        return 0.0
    # 单调化 (从右往左取 max)
    mp = precisions.clone()
    for i in range(mp.numel() - 2, -1, -1):
        mp[i] = max(mp[i].item(), mp[i + 1].item())

    ap = 0.0
    r_points = torch.linspace(0.0, 1.0, 101)
    for t in r_points:
        mask = recalls >= t
        if mask.any():
            ap += mp[mask].max().item() / 101.0
    return ap


def compute_detection_metrics(
    all_results: List[Dict[str, torch.Tensor]],
    iou_thresholds: List[float] = (0.5, 0.75),
    score_threshold: float = 0.05,
) -> Dict[str, float]:
    """单类 (行人) mAP 计算, 兼容 VOC11 与 COCO 101 点插值.

    输入 all_results: 列表, 每项对应一张图, 形如
        {
            "pred_boxes"  : [P, 4] tensor,
            "pred_scores" : [P]    tensor,
            "gt_boxes"    : [G, 4] tensor,
        }

    还会计算 mAP@[0.5:0.05:0.95] (COCO 主指标) 以及每个阈值的 precision/recall.
    """
    device = torch.device("cpu")

    # 先把所有预测按置信度降序合并, 再遍历, 满足 VOC/COCO 标准做法
    all_scores: List[float] = []
    # per iou_threshold: TP / FP lists
    per_iou_tp: Dict[float, List[int]] = {t: [] for t in iou_thresholds}
    per_iou_fp: Dict[float, List[int]] = {t: [] for t in iou_thresholds}

    total_gt = 0
    # 为了 per-image IoU 分析, 也把每张图的 best IoU (score>=0.5 的预测 vs GT) 收集
    per_image_best_iou: List[List[float]] = []

    for res in all_results:
        pb = res["pred_boxes"].to(device)
        ps = res["pred_scores"].to(device)
        gb = res["gt_boxes"].to(device)

        total_gt += gb.shape[0]

        keep = ps >= score_threshold
        pb = pb[keep]
        ps = ps[keep]

        if pb.shape[0] == 0:
            per_image_best_iou.append([0.0] * gb.shape[0])
            continue

        # 按分数降序
        order = torch.argsort(ps, descending=True)
        pb = pb[order]
        ps = ps[order]

        # 记录每张图 score>=0.5 的预测 vs GT best IoU, 便于作图
        vis_mask = ps >= 0.5
        if gb.shape[0] > 0 and vis_mask.any():
            ious_vis = box_iou(pb[vis_mask], gb)
            per_image_best_iou.append(ious_vis.max(dim=0).values.tolist())
        else:
            per_image_best_iou.append([0.0] * gb.shape[0])

        # 针对每个 iou_threshold, 分别做 greedy 匹配
        for thr in iou_thresholds:
            matched = torch.zeros(gb.shape[0], dtype=torch.bool)
            if gb.shape[0] == 0:
                for s in ps:
                    all_scores.append(s.item())
                    per_iou_tp[thr].append(0)
                    per_iou_fp[thr].append(1)
                continue

            ious = box_iou(pb, gb)
            for i in range(pb.shape[0]):
                best_iou, best_j = ious[i].max(0)
                if best_iou.item() >= thr and not matched[best_j]:
                    matched[best_j] = True
                    per_iou_tp[thr].append(1)
                    per_iou_fp[thr].append(0)
                else:
                    per_iou_tp[thr].append(0)
                    per_iou_fp[thr].append(1)

        # scores 只需要收集一次 (与 iou threshold 无关)
        all_scores.extend(ps.tolist())

    result: Dict[str, float] = {"total_gt": int(total_gt)}

    if total_gt == 0 or len(all_scores) == 0:
        for thr in iou_thresholds:
            result[f"AP@{thr}"] = 0.0
        result["mAP@[0.5:0.95]"] = 0.0
        return result

    # scores 列表在不同 iou_threshold 下都是同样顺序 (因为 scores 不依赖阈值)
    scores_tensor = torch.tensor(all_scores)
    order = torch.argsort(scores_tensor, descending=True)

    def _pr(tp_list: List[int], fp_list: List[int]):
        # 注意: tp/fp 的顺序与 all_scores 是一致的, 因为每张图的预测已在 all_scores 中按
        # 分数降序排列, 跨图直接 append, 最后整体再按 scores 全局排序.
        tp = torch.tensor(tp_list, dtype=torch.float32)[order]
        fp = torch.tensor(fp_list, dtype=torch.float32)[order]
        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        recalls = tp_cum / max(total_gt, 1)
        precisions = tp_cum / (tp_cum + fp_cum).clamp(min=1e-9)
        return precisions, recalls

    # AP@0.5 和 AP@0.75 (两种插值都算一下, 主指标报 COCO 101 点)
    for thr in iou_thresholds:
        p, r = _pr(per_iou_tp[thr], per_iou_fp[thr])
        result[f"AP@{thr}"] = _ap_coco(p, r)
        result[f"AP@{thr}_voc11"] = _ap_voc11(p, r)
        result[f"precision@{thr}"] = float(p[-1].item()) if p.numel() else 0.0
        result[f"recall@{thr}"] = float(r[-1].item()) if r.numel() else 0.0

    # mAP @ [0.5 : 0.05 : 0.95]: COCO 主指标
    coco_ious = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    aps = []
    for thr in coco_ious:
        tp_list: List[int] = []
        fp_list: List[int] = []
        # 重新计算 (因为 iou_thresholds 可能不全包含这些)
        for res in all_results:
            pb = res["pred_boxes"].to(device)
            ps = res["pred_scores"].to(device)
            gb = res["gt_boxes"].to(device)
            keep = ps >= score_threshold
            pb = pb[keep]
            ps = ps[keep]
            if pb.shape[0] == 0:
                continue
            o = torch.argsort(ps, descending=True)
            pb = pb[o]; ps = ps[o]

            if gb.shape[0] == 0:
                for _ in ps:
                    tp_list.append(0); fp_list.append(1)
                continue
            matched = torch.zeros(gb.shape[0], dtype=torch.bool)
            ious = box_iou(pb, gb)
            for i in range(pb.shape[0]):
                best_iou, best_j = ious[i].max(0)
                if best_iou.item() >= thr and not matched[best_j]:
                    matched[best_j] = True
                    tp_list.append(1); fp_list.append(0)
                else:
                    tp_list.append(0); fp_list.append(1)
        if len(tp_list) == 0:
            aps.append(0.0); continue
        p, r = _pr(tp_list, fp_list)
        aps.append(_ap_coco(p, r))

    result["mAP@[0.5:0.95]"] = float(sum(aps) / len(aps))
    result["per_iou_ap"] = {str(t): float(a) for t, a in zip(coco_ious, aps)}
    result["per_image_best_iou"] = per_image_best_iou
    return result


def pr_curve_points(
    all_results: List[Dict[str, torch.Tensor]],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.0,
):
    """返回 (precisions, recalls, scores_sorted) 三个 1-D tensor, 用来画 PR 曲线."""
    scores_list: List[float] = []
    tp_list: List[int] = []
    fp_list: List[int] = []
    total_gt = 0

    for res in all_results:
        pb = res["pred_boxes"]
        ps = res["pred_scores"]
        gb = res["gt_boxes"]
        total_gt += gb.shape[0]

        keep = ps >= score_threshold
        pb = pb[keep]; ps = ps[keep]
        if pb.shape[0] == 0:
            continue
        o = torch.argsort(ps, descending=True)
        pb = pb[o]; ps = ps[o]

        if gb.shape[0] == 0:
            for s in ps:
                scores_list.append(s.item()); tp_list.append(0); fp_list.append(1)
            continue
        matched = torch.zeros(gb.shape[0], dtype=torch.bool)
        ious = box_iou(pb, gb)
        for i in range(pb.shape[0]):
            best_iou, best_j = ious[i].max(0)
            if best_iou.item() >= iou_threshold and not matched[best_j]:
                matched[best_j] = True
                tp_list.append(1); fp_list.append(0)
            else:
                tp_list.append(0); fp_list.append(1)
            scores_list.append(ps[i].item())

    if not scores_list:
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), 0

    scores_t = torch.tensor(scores_list)
    tp_t = torch.tensor(tp_list, dtype=torch.float32)
    fp_t = torch.tensor(fp_list, dtype=torch.float32)
    o = torch.argsort(scores_t, descending=True)
    scores_t = scores_t[o]
    tp_t = tp_t[o]; fp_t = fp_t[o]
    tp_cum = torch.cumsum(tp_t, 0)
    fp_cum = torch.cumsum(fp_t, 0)
    recalls = tp_cum / max(total_gt, 1)
    precisions = tp_cum / (tp_cum + fp_cum).clamp(min=1e-9)
    return precisions, recalls, scores_t, total_gt
