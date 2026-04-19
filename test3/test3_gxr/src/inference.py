"""推理与阈值对比, 生成可视化 + 评估报告。

步骤:
    1. 加载 best.pth
    2. 对验证集每张图做前向, 保存:
       - pred_thresh_grid_XX.png   : 阈值扫描 (GT + 多阈值)
       - pred_side_by_side_XX.png  : 默认阈值 0.5 下的 GT vs Pred
       - 原始预测框数量、阈值过滤后数量、IoU
    3. 在整个验证集上做:
       - PR 曲线 (IoU=0.5 / 0.75)
       - 阈值扫描 (P, R, F1, # preds)
       - IoU 直方图
    4. 写 inference_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from dataset import PennFudanDataset, collate_fn, train_val_split
from transforms import build_transforms
from engine import collect_predictions
from metrics import box_iou, compute_detection_metrics, pr_curve_points
from model import build_model
from visualize import (
    save_iou_histogram,
    save_pr_curve,
    save_prediction_grid,
    save_side_by_side,
    save_threshold_sweep,
)


def _threshold_sweep(results, thresholds: List[float], iou_threshold: float = 0.5):
    """对每个 score threshold, 计算 P/R/F1 + 预测数."""
    precisions, recalls, f1s, counts = [], [], [], []
    for thr in thresholds:
        tp = fp = fn = 0
        n_pred = 0
        for res in results:
            pb = res["pred_boxes"]
            ps = res["pred_scores"]
            gb = res["gt_boxes"]
            keep = ps >= thr
            pb = pb[keep]; ps = ps[keep]
            n_pred += pb.shape[0]

            if gb.shape[0] == 0:
                fp += pb.shape[0]
                continue
            if pb.shape[0] == 0:
                fn += gb.shape[0]
                continue

            order = torch.argsort(ps, descending=True)
            pb = pb[order]; ps = ps[order]
            matched = torch.zeros(gb.shape[0], dtype=torch.bool)
            ious = box_iou(pb, gb)
            for i in range(pb.shape[0]):
                best_iou, best_j = ious[i].max(0)
                if best_iou.item() >= iou_threshold and not matched[best_j]:
                    matched[best_j] = True
                    tp += 1
                else:
                    fp += 1
            fn += int((~matched).sum().item())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        precisions.append(p); recalls.append(r); f1s.append(f1); counts.append(n_pred)
    return precisions, recalls, f1s, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=os.path.join(config.CKPT_DIR, "best.pth"))
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=config.VIS_THRESHOLDS)
    parser.add_argument("--num-samples", type=int, default=8,
                        help="逐张可视化前 N 张")
    parser.add_argument("--threads", type=int, default=config.TORCH_THREADS)
    parser.add_argument("--device", default=config.DEVICE, choices=["cuda", "cpu"])
    parser.add_argument("--variant", default=config.MODEL_VARIANT, choices=["v1", "v2"])
    args = parser.parse_args()

    config.make_dirs()
    torch.set_num_threads(args.threads)
    torch.manual_seed(config.SEED)

    # ---------- 重建相同的 val split ----------
    n_imgs = len(os.listdir(os.path.join(config.DATA_ROOT, "PNGImages")))
    _, val_idx = train_val_split(n_imgs, config.VAL_RATIO, config.SEED)
    ds_val = PennFudanDataset(
        config.DATA_ROOT, transforms=build_transforms(train=False),
        indices=val_idx,
    )

    # ---------- 载入 checkpoint ----------
    device = torch.device(args.device)
    model = build_model(
        num_classes=config.NUM_CLASSES,
        variant=args.variant, pretrained=False,
        trainable_backbone_layers=config.TRAINABLE_BACKBONE_LAYERS,
    )
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"[ckpt] 加载 {args.ckpt} "
          f"(epoch={ckpt.get('epoch')}, ap@0.5={ckpt.get('ap', 0):.4f})")
    print(f"[inference] val_size={len(ds_val)}, 逐张可视化前 {args.num_samples}")

    # ---------- 逐张可视化 (阈值网格 + 2x1 对比) ----------
    per_image_stats = []
    num_samples = min(args.num_samples, len(ds_val))
    with torch.no_grad():
        for i in range(num_samples):
            img, target = ds_val[i]
            output = model([img.to(device)])[0]
            pred_boxes = output["boxes"].cpu()
            pred_scores = output["scores"].cpu()

            grid_path = os.path.join(config.PRED_DIR, f"pred_thresh_grid_{i:02d}.png")
            save_prediction_grid(
                img, target["boxes"], pred_boxes, pred_scores,
                thresholds=args.thresholds, save_path=grid_path,
                title_prefix=f"[{i}] ",
            )

            sbs_path = os.path.join(config.PRED_DIR, f"pred_side_by_side_{i:02d}.png")
            save_side_by_side(
                img, target["boxes"], pred_boxes, pred_scores,
                threshold=0.5, save_path=sbs_path,
                title=f"{target.get('filename','')}  (score>=0.5)",
            )

            threshold_counts = {
                f"{thr:.2f}": int((pred_scores >= thr).sum().item())
                for thr in args.thresholds
            }

            best_iou_per_gt = None
            if target["boxes"].shape[0] > 0 and pred_boxes.shape[0] > 0:
                kept = pred_scores >= 0.5
                kb = pred_boxes[kept]
                if kb.shape[0] > 0:
                    iou_mat = box_iou(kb, target["boxes"])
                    best_iou_per_gt = iou_mat.max(dim=0).values.tolist()

            per_image_stats.append({
                "index_in_val": i,
                "filename": target.get("filename", ""),
                "num_gt": int(target["boxes"].shape[0]),
                "num_raw_predictions": int(pred_boxes.shape[0]),
                "top1_score": float(pred_scores[0].item()) if pred_scores.numel() else 0.0,
                "threshold_counts": threshold_counts,
                "iou_per_gt_at_score_0.5": best_iou_per_gt,
            })
            print(f"  [{i}] {target.get('filename','')} "
                  f"GT={per_image_stats[-1]['num_gt']} "
                  f"raw={per_image_stats[-1]['num_raw_predictions']} "
                  f"top1={per_image_stats[-1]['top1_score']:.3f} "
                  f"thr_counts={threshold_counts}")

    # ---------- 整个验证集的指标与曲线 ----------
    loader_val = DataLoader(
        ds_val, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn,
    )
    print("[inference] 全量验证集评估...")
    all_results = collect_predictions(model, loader_val, device)

    full_metrics = compute_detection_metrics(
        all_results,
        iou_thresholds=config.EVAL_IOU_THRESHOLDS,
        score_threshold=config.EVAL_SCORE_THRESHOLD_FOR_VAL,
    )

    # PR 曲线 (两个 IoU)
    pr_data = {}
    for thr in config.EVAL_IOU_THRESHOLDS:
        p, r, s, _gt = pr_curve_points(all_results, iou_threshold=thr, score_threshold=0.01)
        pr_data[f"AP@{thr}"] = (p, r, full_metrics.get(f"AP@{thr}", 0.0))
    save_pr_curve(pr_data, os.path.join(config.FIG_DIR, "pr_curve.png"))

    # IoU 直方图 (score>=0.5 的最佳 IoU per GT)
    ious_flat: List[float] = []
    for row in full_metrics.get("per_image_best_iou", []):
        ious_flat.extend(row)
    save_iou_histogram(
        ious_flat, os.path.join(config.FIG_DIR, "iou_histogram.png"),
        iou_threshold=0.5,
    )

    # 阈值扫描曲线
    sweep_thrs = config.THRESHOLD_SWEEP
    p_list, r_list, f1_list, c_list = _threshold_sweep(
        all_results, sweep_thrs, iou_threshold=0.5,
    )
    save_threshold_sweep(
        sweep_thrs, p_list, r_list, f1_list, c_list,
        os.path.join(config.FIG_DIR, "threshold_sweep.png"),
    )

    # ---------- 落盘 ----------
    pop_keys = ("per_image_best_iou",)
    report = {
        "ckpt": os.path.abspath(args.ckpt),
        "ckpt_epoch": int(ckpt.get("epoch", -1)),
        "ckpt_ap_at_save": float(ckpt.get("ap", 0.0)),
        "thresholds_for_vis": args.thresholds,
        "per_image": per_image_stats,
        "full_metrics": {k: v for k, v in full_metrics.items() if k not in pop_keys},
        "threshold_sweep": {
            "thresholds": sweep_thrs,
            "precision": p_list,
            "recall": r_list,
            "f1": f1_list,
            "num_predictions": c_list,
        },
        "iou_stats": {
            "count": len(ious_flat),
            "mean": float(np.mean(ious_flat)) if ious_flat else 0.0,
            "median": float(np.median(ious_flat)) if ious_flat else 0.0,
            "min": float(min(ious_flat)) if ious_flat else 0.0,
            "max": float(max(ious_flat)) if ious_flat else 0.0,
        },
    }
    report_path = os.path.join(config.OUTPUT_DIR, "inference_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] 推理报告 -> {report_path}")
    print(f"  full_metrics: " + " ".join(
        f"{k}={v:.4f}" for k, v in full_metrics.items()
        if isinstance(v, float) and k not in pop_keys
    ))


if __name__ == "__main__":
    main()
