"""
evaluation.py — 性能评估与可视化

生成图表
--------
  1. confusion_matrices.png    — 三种策略的混淆矩阵（2×3 子图）
  2. roc_curves.png            — DT三策略 + LR基线 ROC 曲线对比
  3. precision_recall_f1.png   — Precision / Recall / F1 柱状图
  4. metrics_heatmap.png       — 分类报告热力图（正/负类 × P/R/F1）
  5. feature_importance.png    — 最优决策树的特征重要性 Top-20
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_score, recall_score, f1_score,
)

from config import OUT_DIR, DPI, PLOT_STYLE, PLOT_PALETTE

os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style=PLOT_STYLE, palette=PLOT_PALETTE)

# 配色方案
COLORS = {
    'DT + SMOTE':       '#2196F3',   # 蓝
    'DT + UnderSample': '#FF5722',   # 橙
    'DT + ClassWeight': '#4CAF50',   # 绿
    'LR Baseline':      '#9C27B0',   # 紫
}


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_all(results: dict, y_test: np.ndarray):
    """
    打印所有模型的 classification_report 并收集指标。

    参数
    ----
    results : {name: {model, y_pred, y_prob, fpr, tpr, auc, cm}}
    y_test  : 真实标签

    返回
    ----
    results : 同输入 dict（原地补充 metrics 字段）
    """
    print("=" * 65)
    print("STEP 6  模型评估（测试集，原始类别分布）")
    print("=" * 65)

    for name, res in results.items():
        y_pred, y_prob = res['y_pred'], res['y_prob']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc_val  = auc(fpr, tpr)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        res.update(fpr=fpr, tpr=tpr, auc=roc_auc_val, cm=cm)
        res['metrics'] = {
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall':    recall_score(y_test, y_pred, zero_division=0),
            'F1-Score':  f1_score(y_test, y_pred, zero_division=0),
        }

        print(f"\n{'─'*60}")
        print(f"  模型: {name}   AUC = {roc_auc_val:.4f}")
        print(f"{'─'*60}")
        print(classification_report(
            y_test, y_pred,
            target_names=['未流失 (No)', '流失 (Yes)'],
            digits=4,
        ))
        print(f"  混淆矩阵: TN={tn}  FP={fp}  FN={fn}  TP={tp}")
        print(f"  误报率(FPR)={fp/(fp+tn):.4f}   漏报率(FNR)={fn/(fn+tp):.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrices(results: dict, y_test: np.ndarray):
    """绘制所有模型的混淆矩阵（1行 × N列）"""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n + 1, 5))
    fig.suptitle(
        'Confusion Matrix — Decision Tree (3 Imbalance Strategies)',
        fontsize=14, fontweight='bold', y=1.02,
    )
    for ax, (name, res) in zip(axes, results.items()):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=res['cm'],
            display_labels=['No (未流失)', 'Yes (流失)'],
        )
        disp.plot(ax=ax, colorbar=True, cmap='Blues')
        ax.set_title(f"{name}\nAUC = {res['auc']:.4f}", fontsize=11, pad=10)
        ax.tick_params(axis='x', labelrotation=15)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'confusion_matrices.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"已保存: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_curves(results: dict, lr_result: dict | None = None):
    """
    绘制 ROC 曲线：DT三策略 + LR基线对比。

    参数
    ----
    results   : DT 三策略结果字典
    lr_result : {'fpr', 'tpr', 'auc'} 或 None
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    # DT 三条线
    for name, res in results.items():
        color = COLORS.get(name, '#333')
        ax.plot(res['fpr'], res['tpr'], color=color, lw=2.5,
                label=f"{name}  (AUC = {res['auc']:.4f})")
        # 在曲线上标注最优阈值点（Youden index）
        j_scores = res['tpr'] - res['fpr']
        best_idx = np.argmax(j_scores)
        ax.scatter(res['fpr'][best_idx], res['tpr'][best_idx],
                   color=color, s=80, zorder=5, marker='o', edgecolors='white')

    # LR 基线
    if lr_result:
        ax.plot(lr_result['fpr'], lr_result['tpr'],
                color=COLORS['LR Baseline'], lw=2, linestyle='--',
                label=f"LR Baseline  (AUC = {lr_result['auc']:.4f})")

    # 随机分类器
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (AUC = 0.5000)')
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color='grey')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    ax.set_title(
        'ROC Curves — Decision Tree vs. Logistic Regression',
        fontsize=14, fontweight='bold',
    )
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)

    # 标注 AUC 区域参考
    ax.text(0.55, 0.15,
            'AUC > 0.80: Good\nAUC > 0.90: Excellent\nAUC > 0.95: Outstanding',
            fontsize=9, color='grey',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    path = os.path.join(OUT_DIR, 'roc_curves.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"已保存: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
def plot_precision_recall_f1(results: dict):
    """绘制 Precision / Recall / F1 分组柱状图"""
    model_names   = list(results.keys())
    metric_labels = ['Precision', 'Recall', 'F1-Score']
    bar_colors    = [COLORS.get(n, '#999') for n in model_names]

    n_models  = len(model_names)
    x         = np.arange(len(metric_labels))
    width     = 0.22

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (name, color) in enumerate(zip(model_names, bar_colors)):
        vals = [results[name]['metrics'][m] for m in metric_labels]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name,
                      color=color, edgecolor='white', linewidth=0.8, alpha=0.88)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.010,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=9.5, fontweight='bold', color='#333')

    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=13)
    ax.set_ylabel('Score (Positive Class: Churn)', fontsize=12)
    ax.set_title(
        'Precision / Recall / F1-Score — Decision Tree (3 Strategies)',
        fontsize=13, fontweight='bold',
    )
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.35, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for thresh, style in [(0.5, ':'), (0.6, '--'), (0.7, '-.')]:
        ax.axhline(thresh, color='grey', linewidth=0.8, linestyle=style, alpha=0.6)
        ax.text(len(metric_labels) - 0.15, thresh + 0.012,
                f'{thresh:.1f}', fontsize=8, color='grey')

    path = os.path.join(OUT_DIR, 'precision_recall_f1.png')
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"已保存: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
def plot_metrics_heatmap(results: dict, y_test: np.ndarray):
    """绘制分类报告热力图（正/负类 × Precision/Recall/F1）"""
    from sklearn.metrics import classification_report as cr

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5))
    fig.suptitle(
        'Classification Report Heatmap — Precision / Recall / F1',
        fontsize=14, fontweight='bold',
    )

    rows_lbl = ['No (未流失)', 'Yes (流失)', 'macro avg', 'weighted avg']
    cols_lbl = ['Precision', 'Recall', 'F1-Score']
    key_map  = {'Precision': 'precision', 'Recall': 'recall', 'F1-Score': 'f1-score'}

    for ax, (name, res) in zip(axes, results.items()):
        report = cr(y_test, res['y_pred'],
                    target_names=['No (未流失)', 'Yes (流失)'],
                    output_dict=True)
        matrix = np.array([
            [report[r][key_map[c]] for c in cols_lbl]
            for r in rows_lbl
        ])
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(cols_lbl)))
        ax.set_xticklabels(cols_lbl, fontsize=10)
        ax.set_yticks(range(len(rows_lbl)))
        ax.set_yticklabels(rows_lbl, fontsize=9)
        ax.set_title(f"{name}\nAUC = {res['auc']:.4f}", fontsize=11, pad=8)

        for r in range(len(rows_lbl)):
            for c in range(len(cols_lbl)):
                v = matrix[r, c]
                ax.text(c, r, f'{v:.3f}', ha='center', va='center',
                        fontsize=10.5, fontweight='bold',
                        color='white' if v < 0.4 or v > 0.82 else 'black')
        plt.colorbar(im, ax=ax, shrink=0.85)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'metrics_heatmap.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"已保存: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(best_model, feature_names: list, strategy_name: str):
    """绘制最优决策树特征重要性 Top-20"""
    importances = best_model.feature_importances_
    idx = np.argsort(importances)[::-1][:20]
    top_names  = [feature_names[i] for i in idx]
    top_values = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_names)))[::-1]
    bars = ax.barh(range(len(top_names)), top_values, color=colors,
                   edgecolor='white', linewidth=0.6)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (Gini/Entropy Gain)', fontsize=11)
    ax.set_title(
        f'Top-20 Feature Importance\n({strategy_name})',
        fontsize=13, fontweight='bold',
    )
    for bar, val in zip(bars, top_values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9, color='#333')
    ax.grid(axis='x', alpha=0.35, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'feature_importance.png')
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"已保存: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
def print_summary(results: dict, lr_auc: float | None = None):
    """打印最终汇总表"""
    print("\n" + "=" * 65)
    print("STEP 8  结果汇总")
    print("=" * 65)
    print(f"\n{'模型':<22} {'测试AUC':>9}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("-" * 65)
    for name, res in results.items():
        m = res['metrics']
        print(f"{name:<22} {res['auc']:>9.4f}  {m['Precision']:>10.4f}  "
              f"{m['Recall']:>8.4f}  {m['F1-Score']:>8.4f}")
    if lr_auc:
        print(f"\n{'LR Baseline (ref)':<22} {lr_auc:>9.4f}  {'—':>10}  {'—':>8}  {'—':>8}")

    best_name = max(results, key=lambda k: results[k]['auc'])
    best_auc  = results[best_name]['auc']
    print(f"\n最优决策树模型 : {best_name}  (AUC = {best_auc:.4f})")
    if lr_auc:
        diff = best_auc - lr_auc
        sign = "+" if diff >= 0 else ""
        print(f"与 LR 基线对比 : {sign}{diff:+.4f}")
    print("\n实验完成！所有输出文件已保存至 output/")
