"""
model.py — 模型训练与超参数调优

决策树核心参数解析
------------------
criterion        : 节点分裂时的不纯度衡量标准
                   'gini'    — 基尼系数，计算更快，倾向于选择频率高的类
                   'entropy' — 信息增益，理论上更精确，适合类别不平衡

max_depth        : 树的最大深度（叶子层数）
                   小  → 树简单，泛化强，但可能欠拟合（bias↑）
                   大  → 树复杂，可完整记忆训练集，但过拟合风险↑（variance↑）
                   None → 完全生长，直到叶子纯净

min_samples_leaf : 叶子节点最少样本数
                   值越大 → 叶子覆盖面积越广 → 泛化越强 → 类似正则化

min_samples_split: 节点进行分裂所需的最少样本数
                   值越大 → 分裂门槛越高 → 树越简单

ccp_alpha        : Cost-Complexity Pruning 剪枝强度
                   0   → 不剪枝（原始树）
                   大  → 剪掉越多子树 → 模型越简单

class_weight     : 少数类权重补偿
                   'balanced' → 权重 = n_samples / (n_classes × np.bincount(y))

调优策略
--------
  GridSearchCV + 5-Fold Stratified CV
  scoring = 'roc_auc'  （不平衡数据下比 accuracy 更具区分力）
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from config import DT_PARAM_GRID, CV_FOLDS, RANDOM_STATE


def _build_param_grid(class_weight):
    """根据 class_weight 生成决策树参数网格"""
    grid = {k: v for k, v in DT_PARAM_GRID.items()}
    grid['class_weight'] = [class_weight]
    return grid


def train_decision_tree(X_train, y_train, class_weight=None, label=''):
    """
    GridSearchCV 训练决策树，返回最佳估计器。

    参数
    ----
    X_train, y_train  : 训练数据（已处理不平衡）
    class_weight      : None 或 'balanced'
    label             : 用于打印的策略标签

    返回
    ----
    best_estimator : DecisionTreeClassifier
    best_params    : dict
    best_cv_auc    : float
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    param_grid = _build_param_grid(class_weight)

    gs = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    print(f"  [{label}]")
    print(f"    最佳参数 : {gs.best_params_}")
    print(f"    CV-AUC   : {gs.best_score_:.4f}")
    print(f"    树深度   : {best.get_depth()}   叶节点数: {best.get_n_leaves()}")
    return best, gs.best_params_, gs.best_score_


def train_logistic_regression(X_train, y_train):
    """
    训练对比用的逻辑回归模型（SMOTE + L2 正则化），
    仅用于 ROC 曲线对比，不作为主模型评估。

    返回
    ----
    best_estimator : LogisticRegression
    """
    print("\n  [LR Baseline — for ROC comparison]")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    param_grid = [
        {'C': [0.01, 0.05, 0.1, 0.5, 1, 5], 'penalty': ['l2'],
         'solver': ['lbfgs'], 'max_iter': [3000]},
        {'C': [0.01, 0.05, 0.1, 0.5, 1],    'penalty': ['l1'],
         'solver': ['liblinear'], 'max_iter': [3000]},
    ]
    gs = GridSearchCV(
        estimator=LogisticRegression(
            random_state=RANDOM_STATE, class_weight='balanced'
        ),
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_train, y_train)
    print(f"    最佳参数 : {gs.best_params_}   CV-AUC: {gs.best_score_:.4f}")
    return gs.best_estimator_
