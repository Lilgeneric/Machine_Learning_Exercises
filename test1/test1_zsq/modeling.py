"""
modeling.py - 超参数调优与模型训练

使用 GridSearchCV + StratifiedKFold(k=5) 对逻辑回归核心参数进行搜索：

  C       : 正则化强度的倒数（越小正则化越强）
              LogisticRegression 的损失函数为：
              min  (1/C) * ||w||^p  +  Σ log(1 + exp(-y_i * w·x_i))
              C 大 → 更信任数据（可能过拟合）
              C 小 → 更依赖正则化（可能欠拟合）

  penalty : 正则化类型
              l1 (Lasso)       → 产生稀疏解，相当于特征选择
              l2 (Ridge)       → 所有特征权重均匀缩小，数值更稳定
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from config import PARAM_GRID, CV_FOLDS, RANDOM_STATE


def _build_base_lr(class_weight=None) -> LogisticRegression:
    """构建基础逻辑回归，class_weight 由调用方传入"""
    return LogisticRegression(
        random_state=RANDOM_STATE,
        class_weight=class_weight,
    )


def run_grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy_name: str,
    class_weight=None,
) -> LogisticRegression:
    """
    对单个策略执行 GridSearchCV。
    评分指标: 'roc_auc'（直接优化 AUC，比 f1 更符合任务目标）
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)

    gs = GridSearchCV(
        estimator  = _build_base_lr(class_weight),
        param_grid = PARAM_GRID,
        cv         = cv,
        scoring    = 'roc_auc',
        n_jobs     = -1,
        verbose    = 0,
        refit      = True,
    )
    gs.fit(X_train, y_train)

    best = gs.best_params_
    print(f"  [{strategy_name}]")
    print(f"    最佳参数: penalty={best.get('penalty')}, "
          f"C={best.get('C')}, solver={best.get('solver')}")
    print(f"    CV-AUC: {gs.best_score_:.4f}")

    return gs.best_estimator_


def train_all_models(strategies: dict) -> dict:
    """
    遍历三种不平衡处理策略，分别做网格搜索并返回最优模型。

    参数:
        strategies: imbalance.get_strategies() 的返回值
                    { 名称: (X_train, y_train, class_weight) }
    返回:
        { 名称: 已拟合的最优 LogisticRegression }
    """
    print("[模型训练] GridSearchCV + 5-Fold CV (scoring=roc_auc)")
    print("-" * 55)
    fitted_models = {}
    for name, (X_tr, y_tr, cw) in strategies.items():
        fitted_models[name] = run_grid_search(X_tr, y_tr, name, cw)
    return fitted_models
