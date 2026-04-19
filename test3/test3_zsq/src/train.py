"""训练入口：读取 config，搭建数据/模型/优化器，训练并保存最佳 checkpoint。

使用：
    cd test3_zsq && python src/train.py            # 默认 config.NUM_EPOCHS
    python src/train.py --epochs 10 --smoke-test   # 快速冒烟测试
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader, Subset

import config
from dataset import PennFudanDataset, build_transforms, collate_fn
from engine import evaluate, train_one_epoch
from model import build_model
from visualize import (save_iter_loss_curve, save_loss_curve, save_map_curve,
                       save_sample_with_gt)


class TeeLogger:
    def __init__(self, log_file: str):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.f = open(log_file, "w", buffering=1)

    def __call__(self, *args, **kwargs):
        msg = " ".join(str(a) for a in args)
        print(msg, flush=True)
        self.f.write(msg + "\n")


def split_dataset(ds, val_ratio: float, seed: int):
    n = len(ds)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    n_val = max(1, int(n * val_ratio))
    return idx[n_val:], idx[:n_val]


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
    args = parser.parse_args()

    config.make_dirs()
    log_file = os.path.join(config.OUTPUT_DIR, "train.log")
    logger = TeeLogger(log_file)

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(config.TORCH_INTEROP_THREADS)
    torch.manual_seed(config.SEED)

    logger(f"[cfg] device={args.device} threads={args.threads} "
           f"epochs={args.epochs} batch={args.batch_size} lr={args.lr}")
    logger(f"[cfg] data_root={config.DATA_ROOT}")

    ds_train = PennFudanDataset(config.DATA_ROOT, transforms=build_transforms(train=True))
    ds_val = PennFudanDataset(config.DATA_ROOT, transforms=build_transforms(train=False))

    train_idx, val_idx = split_dataset(ds_train, config.VAL_RATIO, config.SEED)
    ds_train = Subset(ds_train, train_idx)
    ds_val = Subset(ds_val, val_idx)

    logger(f"[data] total={len(train_idx) + len(val_idx)} train={len(ds_train)} val={len(ds_val)}")

    try:
        img0, tgt0 = ds_train[0]
        sample_path = os.path.join(config.FIG_DIR, "sample_with_gt.png")
        save_sample_with_gt(img0, tgt0, sample_path, title="Penn-Fudan Sample (GT)")
        logger(f"[vis] 样例 GT 图已保存: {sample_path}")
    except Exception as e:
        logger(f"[vis][warn] 样例保存失败: {e}")

    loader_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
        persistent_workers=(config.NUM_WORKERS > 0),
    )
    loader_val = DataLoader(
        ds_val, batch_size=1, shuffle=False,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn,
        persistent_workers=(config.NUM_WORKERS > 0),
    )

    device = torch.device(args.device)
    model = build_model(num_classes=config.NUM_CLASSES, pretrained=True)
    model.to(device)
    logger("[model] fasterrcnn_resnet50_fpn (COCO_V1) → 2 classes (bg + pedestrian)")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY,
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.LR_STEP, gamma=config.LR_GAMMA,
    )

    epoch_avg_losses = []
    iter_losses_all = []
    ap_history = []
    best_ap = -1.0
    best_epoch = -1
    start_time = time.time()

    for epoch in range(args.epochs):
        ep_t0 = time.time()
        history = train_one_epoch(
            model, optimizer, loader_train, device, epoch=epoch,
            print_every=10, warmup=True, logger=logger,
        )

        if args.smoke_test:
            logger("[smoke-test] 首 epoch 完成，提前退出。")
            break

        avg = {k: float(sum(v) / max(len(v), 1)) for k, v in history.items() if k != "lr"}
        epoch_avg_losses.append(avg)
        iter_losses_all.extend(history["loss"])

        lr_scheduler.step()

        metrics = evaluate(
            model, loader_val, device,
            iou_threshold=config.EVAL_IOU_THRESHOLD,
            score_threshold=config.EVAL_SCORE_THRESHOLD,
            logger=logger,
        )
        ap_history.append(metrics["AP"])

        save_loss_curve(epoch_avg_losses, os.path.join(config.FIG_DIR, "loss_curve.png"))
        save_iter_loss_curve(iter_losses_all, os.path.join(config.FIG_DIR, "loss_per_iter.png"))
        save_map_curve(ap_history, os.path.join(config.FIG_DIR, "map_curve.png"))

        ckpt_path = os.path.join(config.CKPT_DIR, "last.pth")
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "ap": metrics["AP"],
        }, ckpt_path)

        if metrics["AP"] > best_ap:
            best_ap = metrics["AP"]
            best_epoch = epoch
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "ap": metrics["AP"],
            }, os.path.join(config.CKPT_DIR, "best.pth"))
            logger(f"[ckpt] 新最佳 mAP@0.5={best_ap:.4f} (epoch {epoch})")

        logger(
            f"[epoch {epoch} 完成] avg_loss={avg['loss']:.4f} "
            f"val_AP={metrics['AP']:.4f} time={time.time()-ep_t0:.1f}s "
            f"elapsed={(time.time()-start_time)/60:.1f}min"
        )

        with open(os.path.join(config.OUTPUT_DIR, "metrics.json"), "w") as f:
            json.dump({
                "epoch_avg_losses": epoch_avg_losses,
                "ap_history": ap_history,
                "best_ap": best_ap,
                "best_epoch": best_epoch,
                "total_epochs_done": epoch + 1,
                "config": {
                    "epochs": args.epochs, "batch_size": args.batch_size,
                    "lr": args.lr, "threads": args.threads,
                },
            }, f, indent=2)

    total_min = (time.time() - start_time) / 60
    logger(f"[done] 训练完成，总耗时 {total_min:.1f} 分钟，最佳 AP@0.5={best_ap:.4f} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
