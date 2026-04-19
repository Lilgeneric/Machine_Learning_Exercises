"""推理与阈值对比脚本。

- 加载 best.pth
- 对验证集若干张图，输出不同置信度阈值下的检测结果（并统计高于阈值的预测框数）
- 生成整体预测统计与 IoU 列表，写入 outputs/inference_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader, Subset

import config
from dataset import PennFudanDataset, build_transforms, collate_fn
from engine import box_iou, evaluate
from model import build_model
from visualize import save_prediction_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=os.path.join(config.CKPT_DIR, "best.pth"))
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=config.VIS_THRESHOLDS)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--threads", type=int, default=config.TORCH_THREADS)
    parser.add_argument("--device", default=config.DEVICE, choices=["cuda", "cpu"])
    args = parser.parse_args()

    config.make_dirs()

    torch.set_num_threads(args.threads)
    torch.manual_seed(config.SEED)

    ds = PennFudanDataset(config.DATA_ROOT, transforms=build_transforms(train=False))
    n = len(ds)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(config.SEED)).tolist()
    n_val = max(1, int(n * config.VAL_RATIO))
    val_idx = idx[:n_val]
    ds_val = Subset(ds, val_idx)

    device = torch.device(args.device)
    model = build_model(num_classes=config.NUM_CLASSES, pretrained=False)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"[ckpt] 加载 {args.ckpt} (epoch={ckpt.get('epoch')}, ap={ckpt.get('ap', 0):.4f})")

    print(f"[inference] 总验证集 {len(ds_val)}；可视化前 {args.num_samples} 张")

    per_image_stats = []
    iou_per_image = []

    with torch.no_grad():
        for i, vi in enumerate(val_idx[:args.num_samples]):
            img, target = ds[vi]
            output = model([img.to(device)])[0]
            pred_boxes = output["boxes"].cpu()
            pred_scores = output["scores"].cpu()

            save_path = os.path.join(config.PRED_DIR, f"pred_thresh_grid_{i:02d}.png")
            save_prediction_grid(
                img, target["boxes"], pred_boxes, pred_scores,
                thresholds=args.thresholds, save_path=save_path,
                title_prefix=f"[{i}] ",
            )

            threshold_counts = {
                f"{thr:.2f}": int((pred_scores >= thr).sum().item())
                for thr in args.thresholds
            }

            iou_mat = None
            best_iou_per_gt = None
            if target["boxes"].shape[0] > 0 and pred_boxes.shape[0] > 0:
                keep = pred_scores >= 0.5
                kept_boxes = pred_boxes[keep]
                if kept_boxes.shape[0] > 0:
                    iou_mat = box_iou(kept_boxes, target["boxes"])
                    best_iou_per_gt = iou_mat.max(dim=0).values.tolist()

            stat = {
                "index": int(vi),
                "num_gt": int(target["boxes"].shape[0]),
                "num_raw_predictions": int(pred_boxes.shape[0]),
                "top1_score": float(pred_scores[0].item()) if pred_scores.numel() else 0.0,
                "threshold_counts": threshold_counts,
                "iou_per_gt_at_0.5_score": best_iou_per_gt,
                "image_file": os.path.basename(save_path),
            }
            per_image_stats.append(stat)
            iou_per_image.append(best_iou_per_gt or [])
            print(f"  [{i}] GT={stat['num_gt']} raw_pred={stat['num_raw_predictions']} "
                  f"thr_counts={threshold_counts}")

    loader_val = DataLoader(
        ds_val, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    threshold_ap = {}
    for thr in args.thresholds:
        m = evaluate(model, loader_val, device,
                     iou_threshold=0.5, score_threshold=thr)
        threshold_ap[f"score>={thr:.2f}"] = m

    report = {
        "ckpt_epoch": int(ckpt.get("epoch", -1)),
        "ckpt_ap_at_save": float(ckpt.get("ap", 0.0)),
        "thresholds": args.thresholds,
        "per_image": per_image_stats,
        "threshold_global_metrics": threshold_ap,
    }

    report_path = os.path.join(config.OUTPUT_DIR, "inference_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[done] 报告写入 {report_path}")


if __name__ == "__main__":
    main()
