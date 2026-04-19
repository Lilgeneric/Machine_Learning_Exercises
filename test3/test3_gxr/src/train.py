"""训练入口。

使用:
    cd test3/test3_gxr && python src/train.py
    python src/train.py --epochs 5 --smoke-test   # 结构验证
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader

import config
from dataset import PennFudanDataset, collate_fn, train_val_split
from transforms import build_transforms
from engine import evaluate, train_one_epoch
from model import build_model, count_parameters
from visualize import (
    save_gt_grid,
    save_iter_loss_curve,
    save_loss_curve,
    save_lr_curve,
    save_map_curve,
    save_sample_with_gt,
)


class TeeLogger:
    """同时写到 stdout 与文件, 用于长时间训练方便事后复盘."""

    def __init__(self, log_file: str):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.f = open(log_file, "w", buffering=1)

    def __call__(self, *args, **kwargs):
        msg = " ".join(str(a) for a in args)
        print(msg, flush=True)
        self.f.write(msg + "\n")

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def seed_everything(seed: int):
    """固定随机种子, 让训练可复现."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--smoke-test", action="store_true",
                        help="只跑 2 个 iter 用于流程验证")
    parser.add_argument("--threads", type=int, default=config.TORCH_THREADS)
    parser.add_argument("--device", default=config.DEVICE,
                        choices=["cuda", "cpu"])
    parser.add_argument("--variant", default=config.MODEL_VARIANT,
                        choices=["v1", "v2"])
    args = parser.parse_args()

    config.make_dirs()
    log_file = os.path.join(config.LOG_DIR, "train.log")
    logger = TeeLogger(log_file)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(config.TORCH_INTEROP_THREADS)
    seed_everything(config.SEED)

    logger(f"[cfg] {config.summary()}")
    logger(f"[cfg] device={args.device} threads={args.threads} "
           f"epochs={args.epochs} batch={args.batch_size} lr={args.lr} "
           f"variant={args.variant}")
    logger(f"[cfg] data_root={config.DATA_ROOT}")

    # ---------- 数据集划分 ----------
    n_imgs = len(os.listdir(os.path.join(config.DATA_ROOT, "PNGImages")))
    train_idx, val_idx = train_val_split(n_imgs, config.VAL_RATIO, config.SEED)
    logger(f"[data] total={n_imgs} train={len(train_idx)} val={len(val_idx)}")

    ds_train = PennFudanDataset(
        config.DATA_ROOT, transforms=build_transforms(train=True),
        indices=train_idx,
    )
    ds_val = PennFudanDataset(
        config.DATA_ROOT, transforms=build_transforms(train=False),
        indices=val_idx,
    )

    # 样例 GT 图 (作业要求之一: 生成数据集中的带 box 标注原图)
    try:
        samples = []
        for i in [0, 1, 2, 3]:
            img, target = ds_train[i]
            samples.append((img, target, f"train/{i}  {target['filename']}"))
        save_gt_grid(
            samples, os.path.join(config.FIG_DIR, "gt_samples.png"), ncols=2,
        )
        for i in [0, 1]:
            img, target = ds_val[i]
            save_sample_with_gt(
                img, target,
                os.path.join(config.FIG_DIR, f"gt_val_{i}.png"),
                title=f"Val GT ({target['filename']})",
            )
        logger(f"[vis] GT 样例已保存: {config.FIG_DIR}/gt_samples.png")
    except Exception as e:
        logger(f"[vis][warn] 样例保存失败: {e}")

    loader_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
        persistent_workers=(config.NUM_WORKERS > 0),
        pin_memory=(args.device == "cuda"),
    )
    loader_val = DataLoader(
        ds_val, batch_size=1, shuffle=False,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
        persistent_workers=(config.NUM_WORKERS > 0),
        pin_memory=(args.device == "cuda"),
    )

    # ---------- 模型 ----------
    device = torch.device(args.device)
    model = build_model(
        num_classes=config.NUM_CLASSES,
        variant=args.variant,
        pretrained=True,
        trainable_backbone_layers=config.TRAINABLE_BACKBONE_LAYERS,
    )
    model.to(device)
    logger(f"[model] fasterrcnn_resnet50_fpn_{args.variant} (COCO) "
           f"-> {config.NUM_CLASSES} classes (bg + pedestrian)")
    logger(f"[model] {count_parameters(model)}")

    # ---------- 优化器 & 调度器 ----------
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr,
        momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY,
    )
    if config.LR_SCHEDULER == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=config.LR_STEP, gamma=config.LR_GAMMA,
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
        )

    # ---------- 训练主循环 ----------
    epoch_avg_losses = []
    iter_losses_all = []
    iter_lrs_all = []
    ap_history = {"AP@0.5": [], "AP@0.75": [], "mAP@[0.5:0.95]": []}
    final_metrics = None
    best_ap = -1.0
    best_epoch = -1
    start_time = time.time()

    for epoch in range(args.epochs):
        ep_t0 = time.time()
        history = train_one_epoch(
            model, optimizer, loader_train, device, epoch=epoch,
            print_every=10, warmup=True,
            warmup_factor=config.WARMUP_FACTOR,
            warmup_iters_cap=config.WARMUP_ITERS_CAP,
            grad_clip_max_norm=config.GRAD_CLIP_MAX_NORM,
            logger=logger,
        )

        if args.smoke_test:
            logger("[smoke-test] 首 epoch 完成, 提前退出.")
            break

        avg = {k: float(sum(v) / max(len(v), 1))
               for k, v in history.items() if k != "lr"}
        epoch_avg_losses.append(avg)
        iter_losses_all.extend(history["loss"])
        iter_lrs_all.extend(history["lr"])

        lr_scheduler.step()

        metrics = evaluate(
            model, loader_val, device,
            iou_thresholds=config.EVAL_IOU_THRESHOLDS,
            score_threshold=config.EVAL_SCORE_THRESHOLD_FOR_VAL,
            logger=logger,
        )
        final_metrics = metrics
        ap_history["AP@0.5"].append(float(metrics.get("AP@0.5", 0.0)))
        ap_history["AP@0.75"].append(float(metrics.get("AP@0.75", 0.0)))
        ap_history["mAP@[0.5:0.95]"].append(float(metrics.get("mAP@[0.5:0.95]", 0.0)))

        # 曲线刷新 (每个 epoch 都覆盖)
        save_loss_curve(
            epoch_avg_losses,
            os.path.join(config.FIG_DIR, "loss_curve.png"),
        )
        save_iter_loss_curve(
            iter_losses_all,
            os.path.join(config.FIG_DIR, "loss_per_iter.png"),
        )
        save_map_curve(
            ap_history, os.path.join(config.FIG_DIR, "map_curve.png"),
        )
        save_lr_curve(
            iter_lrs_all, os.path.join(config.FIG_DIR, "lr_schedule.png"),
        )

        # checkpoint
        ckpt_payload = {
            "model": model.state_dict(),
            "epoch": epoch,
            "ap": metrics.get("AP@0.5", 0.0),
            "variant": args.variant,
        }
        torch.save(ckpt_payload, os.path.join(config.CKPT_DIR, "last.pth"))

        cur_ap = metrics.get("AP@0.5", 0.0)
        if cur_ap > best_ap:
            best_ap = cur_ap
            best_epoch = epoch
            torch.save(ckpt_payload, os.path.join(config.CKPT_DIR, "best.pth"))
            logger(f"[ckpt] 新最佳 AP@0.5={best_ap:.4f} (epoch {epoch})")

        logger(
            f"[epoch {epoch} 完成] avg_loss={avg['loss']:.4f} "
            f"val_AP@0.5={cur_ap:.4f} val_mAP={metrics.get('mAP@[0.5:0.95]', 0):.4f} "
            f"time={time.time()-ep_t0:.1f}s "
            f"elapsed={(time.time()-start_time)/60:.1f}min"
        )

        with open(os.path.join(config.OUTPUT_DIR, "metrics.json"), "w") as f:
            # per_image_best_iou 对 JSON 太大, 这里删掉
            dump_metrics = dict(final_metrics) if final_metrics else {}
            dump_metrics.pop("per_image_best_iou", None)
            json.dump({
                "epoch_avg_losses": epoch_avg_losses,
                "ap_history": ap_history,
                "best_ap": best_ap,
                "best_epoch": best_epoch,
                "total_epochs_done": epoch + 1,
                "last_epoch_metrics": dump_metrics,
                "config": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "variant": args.variant,
                    "seed": config.SEED,
                    "val_ratio": config.VAL_RATIO,
                    "scheduler": config.LR_SCHEDULER,
                    "lr_step": config.LR_STEP,
                },
            }, f, indent=2)

    total_min = (time.time() - start_time) / 60
    logger(f"[done] 训练完成, 总耗时 {total_min:.1f} 分钟, "
           f"最佳 AP@0.5={best_ap:.4f} (epoch {best_epoch})")
    logger.close()


if __name__ == "__main__":
    main()
