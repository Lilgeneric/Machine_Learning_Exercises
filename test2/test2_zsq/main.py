"""
main.py - 主执行入口

运行方式：
    conda run -n ml python main.py

完整流程：
    1. 数据加载（修复 TotalCharges 类型问题）
    2. 特征工程（num_services、monthly_to_total 等 6 个衍生特征）
    3. 特征编码（Binary / Label Encoding / One-Hot Encoding）
    4. 训练/测试划分 + StandardScaler 标准化
    5. 三种类别不平衡处理策略（SMOTE / 欠采样 / class_weight='balanced'）
    6. 决策树 GridSearchCV + 5-Fold CV（优化 AUC）
    7. 逻辑回归基准模型训练（用于 ROC 对比）
    8. 测试集评估（classification_report）
    9. 可视化（指标柱状图 + 混淆矩阵 + ROC 曲线）
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

# ── 本项目模块 ────────────────────────────────────────────────────────────────
from preprocessing import load_raw, feature_engineer, encode, split_scale
from imbalance     import get_strategies
from modeling      import train_all_dt_models, train_lr_baseline
from evaluation    import (
    evaluate_all,
    print_classification,
    plot_metrics_bar,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_feature_importance,
    print_summary,
)


def main():
    print("=" * 65)
    print("  基于决策树算法的电信客户流失预测")
    print("  Telco Customer Churn Prediction with Decision Tree")
    print("=" * 65)

    # ── Step 1: 数据加载 ──────────────────────────────────────────────────────
    print("\n[Step 1] 数据加载")
    df_raw = load_raw()

    # ── Step 2: 特征工程 ──────────────────────────────────────────────────────
    print("\n[Step 2] 特征工程")
    df_feat = feature_engineer(df_raw)

    # ── Step 3: 特征编码 ──────────────────────────────────────────────────────
    print("\n[Step 3] 特征编码（Binary / Label / One-Hot）")
    df_enc = encode(df_feat)

    # ── Step 4: 划分 + 标准化 ─────────────────────────────────────────────────
    print("\n[Step 4] 数据划分 / StandardScaler 标准化")
    X_train, X_test, y_train, y_test, feat_names = split_scale(df_enc)

    # ── Step 5: 类别不平衡处理 ────────────────────────────────────────────────
    print("\n[Step 5] 类别不平衡处理（SMOTE / UnderSample / ClassWeight）")
    strategies = get_strategies(X_train, y_train)

    # ── Step 6: 决策树超参数调优 ──────────────────────────────────────────────
    print("\n[Step 6] 决策树超参数调优（GridSearchCV + 5-Fold CV）")
    dt_models = train_all_dt_models(strategies, X_train, y_train)

    # ── Step 7: 逻辑回归对比模型 ──────────────────────────────────────────────
    print("\n[Step 7] 逻辑回归基准模型（ROC 对比）")
    lr_model = train_lr_baseline(X_train, y_train)

    # 合并所有模型（DT × 3 + LR × 1）
    all_models = {**dt_models, 'LR (ClassWeight)': lr_model}

    # ── Step 8: 测试集评估 ────────────────────────────────────────────────────
    print("\n[Step 8] 测试集评估")
    results = evaluate_all(all_models, X_test, y_test)
    print_classification(results, y_test)

    # ── Step 9: 可视化 ────────────────────────────────────────────────────────
    print("\n[Step 9] 可视化图表生成")
    plot_metrics_bar(results)
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results)
    plot_feature_importance(dt_models, feat_names)

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    print_summary(results)

    print("\n" + "=" * 65)
    print("  实验完成！所有图表已保存至 output/ 目录")
    print("=" * 65)


if __name__ == '__main__':
    main()
