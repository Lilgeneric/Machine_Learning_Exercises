"""
main.py — 电信客户流失预测（决策树）入口

题目：基于决策树算法的电信客户流失预测
数据集：Telco Customer Churn (Kaggle / IBM)
作者：gxr   日期：2026-03-20   环境：conda ml

运行：
    conda run -n ml python main.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 确保当前目录在 sys.path（支持相对导入）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.metrics import roc_curve, auc

from config      import OUT_DIR, RANDOM_STATE
from preprocessing import load_and_preprocess
from imbalance   import apply_imbalance_strategies
from model       import train_decision_tree, train_logistic_regression
from evaluation  import (
    evaluate_all,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_precision_recall_f1,
    plot_metrics_heatmap,
    plot_feature_importance,
    print_summary,
)

os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("\n" + "█" * 65)
    print("  电信客户流失预测  ——  决策树 + GridSearchCV + 不平衡处理")
    print("█" * 65 + "\n")

    # ── STEP 1-3  数据加载、清洗、特征工程、标准化 ────────────────────────────
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess()

    # ── STEP 4  类别不平衡处理 ────────────────────────────────────────────────
    strategies = apply_imbalance_strategies(X_train, y_train)

    # ── STEP 5  超参数调优（GridSearchCV + 5-Fold Stratified CV）─────────────
    print("\n" + "=" * 65)
    print("STEP 5  超参数调优 (GridSearchCV + 5-Fold Stratified CV)")
    print("=" * 65)
    print("""
[决策树核心参数]
  criterion         : gini / entropy  —— 衡量节点分裂纯度的标准
  max_depth         : 树的最大深度，控制模型复杂度（防过拟合）
  min_samples_leaf  : 叶节点最少样本，起正则化作用（值大→树简单）
  min_samples_split : 节点分裂门槛（值大→分裂保守）
  ccp_alpha         : 成本复杂度剪枝强度（0=不剪枝）
  class_weight      : 少数类权重（'balanced' 自动补偿不平衡）
  scoring           : roc_auc —— 不平衡场景下优于 accuracy
""")

    dt_models = {}   # {strategy_name: (best_model, best_params, cv_auc)}
    for name, (X_tr, y_tr, cw) in strategies.items():
        print(f"\n  训练策略: {name}")
        model, params, cv_auc = train_decision_tree(X_tr, y_tr, class_weight=cw, label=name)
        dt_models[name] = (model, params, cv_auc)

    # ── 对比模型：LR（使用 SMOTE 处理后的训练集）─────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 5b  对比模型：逻辑回归（LR Baseline）")
    print("=" * 65)
    X_tr_smote, y_tr_smote, _ = strategies['DT + SMOTE']
    lr_model = train_logistic_regression(X_tr_smote, y_tr_smote)

    # ── STEP 6  在测试集上生成预测 ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 6  在测试集上预测")
    print("=" * 65)

    results = {}
    for name, (model, params, cv_auc) in dt_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        results[name] = dict(
            model=model, y_pred=y_pred, y_prob=y_prob,
            cv_auc=cv_auc, best_params=params,
        )

    # LR 基线评估
    lr_prob = lr_model.predict_proba(X_test)[:, 1]
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
    lr_auc = auc(lr_fpr, lr_tpr)
    lr_result = {'fpr': lr_fpr, 'tpr': lr_tpr, 'auc': lr_auc}
    print(f"  LR Baseline  测试 AUC = {lr_auc:.4f}")

    # ── STEP 6 (评估)  classification_report + 指标收集 ───────────────────────
    results = evaluate_all(results, y_test)

    # ── STEP 7  可视化 ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 7  可视化输出")
    print("=" * 65)

    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, lr_result)
    plot_precision_recall_f1(results)
    plot_metrics_heatmap(results, y_test)

    # 选取测试 AUC 最高的策略绘制特征重要性
    best_name  = max(results, key=lambda k: results[k]['auc'])
    best_model = results[best_name]['model']
    plot_feature_importance(best_model, feature_names, best_name)

    # ── STEP 8  汇总 ──────────────────────────────────────────────────────────
    print_summary(results, lr_auc)

    print(f"\n输出文件一览 ({OUT_DIR}/):")
    for fname in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, fname)
        size  = os.path.getsize(fpath)
        print(f"  {fname:<35} {size/1024:>7.1f} KB")


if __name__ == '__main__':
    main()
