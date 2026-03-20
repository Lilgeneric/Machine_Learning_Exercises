"""
evaluation.py - 模型评估与可视化

包含：
  evaluate_all()            → 测试集推断，收集 Precision/Recall/F1/AUC 等指标
  print_classification()    → 打印 classification_report（含不平衡指标分析）
  plot_metrics_bar()        → Precision / Recall / F1 分组柱状图
  plot_confusion_matrices() → 混淆矩阵对比图（含 FP/FN 标注说明）
  plot_roc_curves()         → ROC 曲线对比图（决策树 3 种策略 + LR 基准）
  print_summary()           → AUC 汇总与结论

指标说明：
  Precision(流失) = TP/(TP+FP)  ← 预测为"流失"的客户中，真正流失的比例
  Recall(流失)    = TP/(TP+FN)  ← 所有真实流失客户中，被模型找到的比例（业务核心）
  F1-Score        = 2·P·R/(P+R)  ← Precision 与 Recall 的调和平均
  AUC             = ROC 曲线下面积，衡量模型对流失/不流失的整体排序能力
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    accuracy_score,
)

from config import OUT_DIR

os.makedirs(OUT_DIR, exist_ok=True)

# 固定颜色方案：决策树三策略 + LR 对比
COLORS = {
    'DT (SMOTE)':       '#2196F3',   # 蓝色
    'DT (UnderSample)': '#FF9800',   # 橙色
    'DT (ClassWeight)': '#4CAF50',   # 绿色
    'LR (ClassWeight)': '#E91E63',   # 粉红色（LR 对比线）
}

LINE_STYLES = {
    'DT (SMOTE)':       '-',
    'DT (UnderSample)': '--',
    'DT (ClassWeight)': '-.',
    'LR (ClassWeight)': ':',
}


# ── 核心评估函数 ───────────────────────────────────────────────────────────────

def evaluate_all(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    对所有模型（DT×3 + LR×1）在测试集上推断并收集结构化结果。

    返回格式：
        {
          模型名: {
            'y_pred'   : ndarray,
            'y_prob'   : ndarray,     # 正类（流失）概率
            'cm'       : ndarray,     # 混淆矩阵 [[TN,FP],[FN,TP]]
            'fpr'      : ndarray,
            'tpr'      : ndarray,
            'auc'      : float,
            'precision': float,       # 流失类(1)的 Precision
            'recall'   : float,       # 流失类(1)的 Recall（业务最关键）
            'f1'       : float,       # 流失类(1)的 F1
            'accuracy' : float,
          }
        }
    """
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        report = classification_report(
            y_test, y_pred,
            target_names=['未流失 (0)', '流失 (1)'],
            output_dict=True,
        )
        results[name] = {
            'y_pred'   : y_pred,
            'y_prob'   : y_prob,
            'cm'       : confusion_matrix(y_test, y_pred),
            'fpr'      : fpr,
            'tpr'      : tpr,
            'auc'      : auc(fpr, tpr),
            'precision': report['流失 (1)']['precision'],
            'recall'   : report['流失 (1)']['recall'],
            'f1'       : report['流失 (1)']['f1-score'],
            'accuracy' : accuracy_score(y_test, y_pred),
        }
    return results


# ── 分类报告 ───────────────────────────────────────────────────────────────────

def print_classification(results: dict, y_test: np.ndarray) -> None:
    """
    打印每个模型的完整 classification_report。

    重点说明：
      - 在类别不平衡场景下，不能只看 Accuracy（"全部预测为未流失"也能获得 73%+）
      - 业务关键指标：流失类(1)的 Recall（漏掉流失客户代价极高）
      - F1 是 Precision 与 Recall 的平衡，避免单边极端化
    """
    print("\n" + "=" * 65)
    print("  分类报告 (Precision / Recall / F1-Score)")
    print("  ⚠ 重点关注流失类(1)的 Recall，漏报代价高于误报")
    print("=" * 65)
    for name, res in results.items():
        print(f"\n{'─'*55}")
        print(f"  模型: {name}  |  AUC = {res['auc']:.4f}  "
              f"|  Accuracy = {res['accuracy']:.4f}")
        print(classification_report(
            y_test, res['y_pred'],
            target_names=['未流失 (0)', '流失 (1)'],
        ))


# ── 指标柱状图 ─────────────────────────────────────────────────────────────────

def plot_metrics_bar(results: dict) -> str:
    """
    绘制 Precision / Recall / F1-Score / Accuracy 四指标分组柱状图。

    左图：四指标 × 所有模型的分组柱状图（宏观对比）
    右图：流失类(1)的 Precision-Recall-F1 水平条形图（精细对比）

    图表解读：
      Recall(1) 柱最高的模型 → 对流失客户的识别能力最强（业务最优先）
      F1(1)     柱最高的模型 → 综合表现最佳
      Accuracy  仅供参考，不平衡数据下参考价值有限
    """
    models    = list(results.keys())
    colors    = [COLORS.get(m, '#999999') for m in models]
    metrics   = ['Precision(1)', 'Recall(1)', 'F1-Score(1)', 'Accuracy']
    keys      = ['precision', 'recall', 'f1', 'accuracy']
    n_metrics = len(metrics)
    n_models  = len(models)
    bar_width = 0.18
    x         = np.arange(n_metrics)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        'Decision Tree vs LR – Precision / Recall / F1 Comparison\n'
        '(Class=1: Churn，关注流失类指标)',
        fontsize=13, fontweight='bold', y=1.01,
    )

    # ── 左图：分组竖向柱状图 ──────────────────────────────────────────────────
    ax = axes[0]
    for i, (name, color) in enumerate(zip(models, colors)):
        vals   = [results[name][k] for k in keys]
        offset = (i - (n_models - 1) / 2) * bar_width
        bars   = ax.bar(x + offset, vals, bar_width,
                        label=name, color=color, alpha=0.85, edgecolor='white',
                        linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.007,
                f'{v:.3f}',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold',
                color='#333333',
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Four Metrics Comparison (All Models)', fontsize=11)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.85)
    ax.axhline(y=0.5, color='gray', lw=0.8, linestyle='--', alpha=0.4)
    ax.grid(axis='y', alpha=0.25)

    # ── 右图：流失类 P/R/F1 水平条形图（突出权衡关系）─────────────────────────
    ax2 = axes[1]
    pr_metrics = ['Precision (1)', 'Recall (1)', 'F1-Score (1)']
    pr_keys    = ['precision', 'recall', 'f1']
    y_pos      = np.arange(len(pr_metrics))
    bar_h      = 0.18

    for i, (name, color) in enumerate(zip(models, colors)):
        vals   = [results[name][k] for k in pr_keys]
        offset = (i - (n_models - 1) / 2) * bar_h
        bars   = ax2.barh(y_pos + offset, vals, bar_h,
                          label=name, color=color, alpha=0.85,
                          edgecolor='white', linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax2.text(
                v + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}',
                va='center', ha='left', fontsize=8, fontweight='bold',
                color='#333333',
            )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(pr_metrics, fontsize=11)
    ax2.set_xlim(0, 1.20)
    ax2.set_xlabel('Score', fontsize=11)
    ax2.set_title('Precision–Recall–F1 Tradeoff (Class=1: Churn)', fontsize=11)
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.85)
    ax2.axvline(x=0.5, color='gray', lw=0.8, linestyle='--', alpha=0.4)
    ax2.grid(axis='x', alpha=0.25)

    fig.text(
        0.5, -0.03,
        '注：Class=1 为正类（客户流失）。Accuracy 在不平衡数据中参考意义有限，'
        '业务上应优先关注 Recall(1)（减少漏报）和 F1(1)（综合表现）',
        ha='center', fontsize=9, color='#666666',
    )

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'metrics_bar.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 指标柱状图 → {path}")
    return path


# ── 混淆矩阵 ───────────────────────────────────────────────────────────────────

def plot_confusion_matrices(
    results: dict,
    y_test:  np.ndarray,
) -> str:
    """
    绘制所有模型的混淆矩阵对比图（1行 × N列）。

    混淆矩阵四象限：
      TN (左上): 正确预测"未流失"  ← 模型识别出忠实客户
      FP (右上): 误报（把未流失预测为流失）← 浪费客服资源
      FN (左下): 漏报（把流失预测为未流失）← 业务损失最大，应尽量减小
      TP (右下): 正确预测"流失"   ← 成功识别目标客户

    高 Recall 目标 → FN 越小越好（左下格数字越小越好）
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n + 1, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        'Confusion Matrix Comparison – Decision Tree (3 Strategies) + LR\n'
        '目标: 左下(FN)越小越好 ← 减少流失客户漏报',
        fontsize=13, fontweight='bold', y=1.02,
    )

    for ax, (name, res) in zip(axes, results.items()):
        disp = ConfusionMatrixDisplay(
            confusion_matrix = res['cm'],
            display_labels   = ['未流失(0)', '流失(1)'],
        )
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        # 标注 AUC、Recall、F1
        ax.set_title(
            f"{name}\n"
            f"AUC={res['auc']:.4f}  Recall={res['recall']:.3f}  F1={res['f1']:.3f}",
            fontsize=10,
        )
        # 标注 FN 位置
        ax.text(
            -0.3, 1.05, '← FN: 漏报流失',
            transform=ax.transAxes, fontsize=8, color='red', alpha=0.7,
        )

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'confusion_matrices.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 混淆矩阵 → {path}")
    return path


# ── ROC 曲线 ───────────────────────────────────────────────────────────────────

def plot_roc_curves(results: dict) -> str:
    """
    绘制所有模型的 ROC 曲线对比图（含 LR 基准线）。

    ROC 曲线说明：
      x 轴 FPR = FP/(FP+TN)   假阳率（把未流失错判为流失的比例）
      y 轴 TPR = TP/(TP+FN)   真阳率/召回率（成功识别流失客户的比例）

    AUC 含义：
      AUC = 随机选一个流失和一个未流失客户，模型给流失客户更高概率的概率
      AUC = 1.0: 完美分类
      AUC = 0.5: 等同随机猜测
      一般 AUC > 0.85 视为优秀，AUC > 0.90 视为非常好

    DT vs LR 对比思考：
      - LR 是线性模型，只能学习线性决策边界
      - 决策树能捕捉非线性规律（如"高月费 AND 短合约"的组合风险）
      - 在特征工程充分时，DT 通常能超越 LR，尤其是 AUC 维度
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        'ROC Curve Comparison – Decision Tree (3 Strategies) vs LR\n'
        '（AUC 越大越好，曲线越靠近左上角越优秀）',
        fontsize=13, fontweight='bold', y=1.01,
    )

    # ── 左图：全部模型 ROC ────────────────────────────────────────────────────
    ax = axes[0]
    for name, res in results.items():
        color = COLORS.get(name, '#999999')
        ls    = LINE_STYLES.get(name, '-')
        lw    = 3.0 if 'LR' not in name else 2.0
        ax.plot(
            res['fpr'], res['tpr'],
            color=color, lw=lw, linestyle=ls,
            label=f"{name}  (AUC = {res['auc']:.4f})",
        )

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random (AUC = 0.50)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall / TPR)', fontsize=12)
    ax.set_title('Full ROC Curve Comparison', fontsize=12)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.25)

    # ── 右图：放大 FPR 0~0.4 的关键区域 ─────────────────────────────────────
    ax2 = axes[1]
    for name, res in results.items():
        color = COLORS.get(name, '#999999')
        ls    = LINE_STYLES.get(name, '-')
        lw    = 3.0 if 'LR' not in name else 2.0
        ax2.plot(
            res['fpr'], res['tpr'],
            color=color, lw=lw, linestyle=ls,
            label=f"{name}  (AUC = {res['auc']:.4f})",
        )

    ax2.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
    ax2.set_xlim([0.0, 0.4])
    ax2.set_ylim([0.4, 1.02])
    ax2.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax2.set_ylabel('True Positive Rate (Recall / TPR)', fontsize=12)
    ax2.set_title('ROC Zoom-In (FPR ∈ [0, 0.4]) – 关键区域放大', fontsize=12)
    ax2.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax2.grid(alpha=0.25)

    # 标注"理想点"(0,1)
    ax2.annotate(
        '理想点 (0,1)',
        xy=(0, 1), xytext=(0.05, 0.92),
        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
        fontsize=10, color='red',
    )

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'roc_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] ROC 曲线 → {path}")
    return path


# ── AUC 汇总 ───────────────────────────────────────────────────────────────────

AUC_GOOD   = 0.80   # 良好阈值
AUC_GREAT  = 0.85   # 优秀阈值


def plot_feature_importance(
    models:        dict,
    feature_names: list,
    top_n:         int = 20,
) -> str:
    """
    绘制决策树最优模型（按 AUC 排序）的特征重要性 Top-N 横向柱状图。

    特征重要性（Gini Importance）：
      每个特征在树所有分裂中降低基尼不纯度的加权平均贡献。
      值越高 → 该特征对流失预测越关键。
      注意：Gini Importance 会对高基数特征略有偏差，
             但对本数据集的类别特征仍具有很好的参考价值。
    """
    from sklearn.tree import DecisionTreeClassifier

    # 找出 AUC 最高的 DT 模型
    dt_models = {n: m for n, m in models.items()
                 if isinstance(m, DecisionTreeClassifier)}
    if not dt_models:
        return ''

    # 这里我们直接用传入的 models dict 来获取，需要外部传入 results 中最优模型
    # 改为绘制所有 DT 模型的重要性对比
    dt_items = list(dt_models.items())
    n_models = len(dt_items)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 8))
    if n_models == 1:
        axes = [axes]

    fig.suptitle(
        f'Decision Tree Feature Importance (Top {top_n})\n'
        '特征重要性反映了各特征在树分裂中降低基尼不纯度的贡献',
        fontsize=13, fontweight='bold', y=1.01,
    )

    for ax, (name, model) in zip(axes, dt_items):
        importances = model.feature_importances_
        # 取 Top N
        sorted_idx = np.argsort(importances)[-top_n:]
        feat_labels = [feature_names[i] for i in sorted_idx]
        feat_vals   = importances[sorted_idx]
        colors      = [COLORS.get(name, '#2196F3')] * len(feat_vals)
        # 渐变色强调重要特征
        norm_vals   = feat_vals / feat_vals.max()
        bar_colors  = plt.cm.Blues(0.35 + 0.55 * norm_vals)

        bars = ax.barh(
            range(len(feat_labels)), feat_vals,
            color=bar_colors, edgecolor='white', linewidth=0.6,
        )
        ax.set_yticks(range(len(feat_labels)))
        ax.set_yticklabels(feat_labels, fontsize=9)
        ax.set_xlabel('Gini Importance', fontsize=10)
        ax.set_title(f'{name}', fontsize=11)
        ax.grid(axis='x', alpha=0.25)

        # 标注数值
        for bar, v in zip(bars, feat_vals):
            ax.text(
                v + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f'{v:.4f}',
                va='center', ha='left', fontsize=7.5,
            )

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 特征重要性 → {path}")
    return path


def print_summary(results: dict) -> None:
    """
    打印所有模型的 AUC / Precision / Recall / F1 汇总表，
    并给出 DT vs LR 的对比结论。
    """
    print("\n" + "=" * 72)
    print("  性能汇总表")
    print("=" * 72)
    fmt = "  {:<22} {:>7} {:>10} {:>9} {:>9}   {}"
    print(fmt.format('模型', 'AUC', 'Precision', 'Recall', 'F1', '评级'))
    print("  " + "-" * 68)

    best_auc_name = max(results, key=lambda k: results[k]['auc'])

    for name, res in results.items():
        v   = res['auc']
        tag = "★ 最优" if name == best_auc_name else ""
        if v >= AUC_GREAT:
            level = "优秀"
        elif v >= AUC_GOOD:
            level = "良好"
        else:
            level = "一般"
        print(fmt.format(
            name,
            f"{v:.4f}",
            f"{res['precision']:.4f}",
            f"{res['recall']:.4f}",
            f"{res['f1']:.4f}",
            f"{level} {tag}",
        ))

    print("  " + "-" * 68)

    # DT vs LR 分析
    dt_names = [n for n in results if 'DT' in n]
    lr_names = [n for n in results if 'LR' in n]
    if dt_names and lr_names:
        best_dt_auc = max(results[n]['auc'] for n in dt_names)
        best_lr_auc = max(results[n]['auc'] for n in lr_names)
        diff = best_dt_auc - best_lr_auc
        print(f"\n  [DT vs LR 对比]")
        print(f"    最优 DT AUC:    {best_dt_auc:.4f}")
        print(f"    LR AUC:         {best_lr_auc:.4f}")
        if diff > 0:
            print(f"    决策树 AUC 领先 LR: +{diff:.4f}")
            print("    → 树模型能捕捉非线性交互特征（如高月费×短合约的组合风险），")
            print("      在流失预测场景中通常优于线性 LR 模型。")
        else:
            print(f"    LR AUC 领先决策树: +{abs(diff):.4f}")
            print("    → 当前数据集特征工程后线性可分性较强，LR 也能取得竞争力。")
            print("      可考虑使用集成方法（Random Forest / XGBoost）进一步提升。")

    print(f"\n  输出图表目录: {OUT_DIR}/")
    print("    ├─ metrics_bar.png        指标柱状图")
    print("    ├─ confusion_matrices.png 混淆矩阵对比")
    print("    └─ roc_curves.png         ROC 曲线对比（含 LR 基准）")
