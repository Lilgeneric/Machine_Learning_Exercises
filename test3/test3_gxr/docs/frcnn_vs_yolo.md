# Faster R-CNN vs YOLO: 为什么前者慢但通常更精准

## TL;DR

| 维度          | Faster R-CNN      | YOLO (v5/v8)     |
| ------------- | ----------------- | ---------------- |
| 阶段          | 二阶段 (two-stage) | 单阶段 (one-stage) |
| 典型 FPS (V100) | 7-15             | 30-150           |
| COCO mAP     | 46.7 (v2)          | 50+ (YOLOv8-X)   |
| 定位精度      | 高 (RoIAlign)     | 中                |
| 小物体检测    | 较好 (FPN + RoIAlign) | 较差 (grid 限制)  |
| 工程易用性    | torchvision 官方  | 需 ultralytics    |

二者都好, 在不同应用场景各有优势.

## 架构差异

### Faster R-CNN (二阶段)

```
图片 → Backbone → FPN
           ↓
         RPN (第 1 阶段)
           ↓ 候选框 (top-K + NMS)
       RoIAlign (对齐特征)
           ↓
       Box Head (第 2 阶段)  → (cls, box)
```

特点: **先粗提候选, 再精准分类 + 回归**.

### YOLO (单阶段)

```
图片 → Backbone → Neck (PAN/FPN)
                ↓
            Head (cls + box + obj in one tensor)
                ↓
             NMS → 输出
```

特点: **grid 的每个 cell 直接预测 "有没有 + 是什么 + 在哪里"**.

## 为什么 Faster R-CNN 更精准

1. **RoIAlign 消除量化误差**
   RoIPool 把 RoI 量化到整数像素格, 会丢失 0.5 像素级别的精度; RoIAlign 用双线性插值, 对小目标和精细定位非常友好.
2. **候选框质量更高**
   RPN 是学出来的候选生成器, 训练目标就是 "高质量 anchor", 比 YOLO grid 上的均匀先验更贴合数据分布.
3. **两阶段校正**
   第 1 阶段只需要判断 "前景 / 背景" 并粗略 refine 位置; 第 2 阶段接 RPN 的粗框再做一次分类 + 回归. 每个子任务都比 YOLO 的 "一次性什么都预测" 简单.
4. **类别不平衡处理更细致**
   RPN 可以单独采样正负样本比 (1:1); Box Head 又可以二次采样. YOLO 的正样本定义则依赖 grid + IoU 规则, 易出现严重不平衡.

## 为什么 YOLO 更快

1. **一次 forward**: 省去 RPN → RoIAlign → Box Head 的管线.
2. **Anchor-free / grid-based 推理简洁**: 可直接并行化.
3. **无 RoI-level 的循环**: Faster R-CNN 每个 RoI 都要单独跑 head, 计算量线性于 RoI 数.

## 应用场景对比

| 场景                           | 推荐         | 原因                           |
| ------------------------------ | ------------ | ------------------------------ |
| 自动驾驶 (实时)                 | YOLO / DETR   | 延迟 < 30ms 的硬要求           |
| 安防视频流多路                  | YOLO         | 单 GPU 要跑 8-16 路           |
| 医学影像 / 工业瑕疵检测         | Faster R-CNN / Cascade R-CNN | 定位精度 > 速度; 小目标多 |
| 遥感目标检测                    | Faster R-CNN + FPN | 尺度差异大, FPN 必不可少    |
| 移动端 / 嵌入式                 | YOLOv5n / YOLOv8n | 模型小, INT8 部署方便     |
| 学术 benchmark / 小数据集       | Faster R-CNN | 迁移学习效果稳定, 论文对比多  |

## 本实验的选择

Penn-Fudan 只有 170 张图, 单 GPU 训练 2 分钟就能到 mAP@[0.5:0.95] = 0.885, **延迟不是瓶颈, 精度更重要**, 因此选 Faster R-CNN.

同等的 YOLOv8n 在这个数据集上可能训练更快 (几分钟), 但 AP@0.5 大概率也能到 0.95+, 只是 mAP@[0.5:0.95] 会略低于 Faster R-CNN — 原因正是上面讲的 RoIAlign + 二阶段精修.

## 延伸阅读

- *Faster R-CNN* (Ren et al., NeurIPS 2015).
- *YOLOv1* (Redmon et al., CVPR 2016) → *YOLOv8* (Ultralytics, 2023).
- *Mask R-CNN* (He et al., ICCV 2017) — Faster R-CNN 的实例分割扩展, RoIAlign 首次提出.
- *DETR* (Carion et al., ECCV 2020) — 抛弃 anchor / NMS 的新范式, 训练慢但无 hand-crafted 组件.
