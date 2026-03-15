"""
main.py - 主执行入口

运行方式：
    conda run -n ml python main.py

完整流程：
    1. 数据加载
    2. 特征工程（pdays 二值化、对数变换、年龄分段、经济压力指标）
    3. 编码（Label Encoding + One-Hot Encoding）
    4. 训练/测试划分 + 标准化 + 多项式特征扩展
    5. 三种不平衡处理策略（SMOTE / 欠采样 / class_weight）
    6. GridSearchCV + 5-Fold 超参数调优
    7. 测试集评估（classification_report）
    8. 可视化（混淆矩阵 + ROC 曲线）
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

# ── 本项目模块 ────────────────────────────────────────────────────────────────
from preprocessing import load_raw, feature_engineer, encode, split_scale
from imbalance     import get_strategies
from modeling      import train_all_models
from evaluation    import (
    evaluate_all,
    print_classification,
    plot_confusion_matrices,
    plot_roc_curves,
    print_summary,
)


def main():
    print("=" * 60)
    print("  基于逻辑回归的银行营销结果预测")
    print("=" * 60)

    # ── Step 1: 数据加载 ──────────────────────────────────────────────────────
    print("\n[Step 1] 数据加载")
    df_raw = load_raw()

    # ── Step 2: 特征工程 ──────────────────────────────────────────────────────
    print("\n[Step 2] 特征工程")
    df_feat = feature_engineer(df_raw)

    # ── Step 3: 编码 ──────────────────────────────────────────────────────────
    print("\n[Step 3] 特征编码")
    df_enc = encode(df_feat)
    print(f"  编码后特征数: {df_enc.shape[1] - 1}")

    # ── Step 4: 划分 + 缩放 + 多项式展开 ─────────────────────────────────────
    print("\n[Step 4] 数据划分 / 标准化 / 多项式特征")
    X_train, X_test, y_train, y_test, feat_names = split_scale(df_enc)

    # ── Step 5: 类别不平衡处理 ────────────────────────────────────────────────
    print("\n[Step 5] 类别不平衡处理")
    strategies = get_strategies(X_train, y_train)

    # ── Step 6: 超参数调优 ────────────────────────────────────────────────────
    print("\n[Step 6] 超参数调优 (GridSearchCV + 5-Fold CV)")
    models = train_all_models(strategies)

    # ── Step 7: 评估 ──────────────────────────────────────────────────────────
    print("\n[Step 7] 模型评估")
    results = evaluate_all(models, X_test, y_test)
    print_classification(results, y_test)

    # ── Step 8: 可视化 ────────────────────────────────────────────────────────
    print("\n[Step 8] 可视化")
    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results)

    # ── 汇总 ──────────────────────────────────────────────────────────────────
    print_summary(results)


if __name__ == '__main__':
    main()
