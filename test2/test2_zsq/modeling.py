"""
modeling.py - 超参数调优与模型训练

决策树核心参数说明：
  max_depth        : 树的最大深度
                     控制模型复杂度的最重要参数。
                     过浅（<3）→ 欠拟合，无法捕捉数据中的非线性结构
                     过深（>15）→ 过拟合，在训练集表现优异但测试集泛化差
                     本实验搜索范围: [3, 5, 7, 10, 15, 20, None]

  min_samples_leaf : 叶子节点最少样本数（本作业重点）
                     等价于"剪枝"：叶子越小 → 树越细致 → 越易过拟合
                     1  → 允许每片叶子仅包含1个样本（极度过拟合）
                     50 → 叶子至少50个样本（大幅剪枝，稳健性更强）
                     本实验搜索范围: [1, 2, 5, 10, 20]

  min_samples_split: 内部节点拆分所需的最小样本数
                     当节点样本数 < min_samples_split 时停止分裂
                     本实验搜索范围: [2, 5, 10]

  criterion        : 特征分裂的信息度量
                     gini    : 基尼不纯度 = 1 - Σp_i²，计算效率更高
                     entropy : 信息增益 = -Σp_i·log(p_i)，对小概率类更敏感
                     本实验同时搜索两者

网格规模：7 × 5 × 3 × 2 = 210 组参数 × 5折 = 1050 次拟合/策略
使用 n_jobs=-1 并行计算。

逻辑回归（LR）对比模型：
  作为 ROC 曲线比较的基准，使用 GridSearchCV 独立调优 LR，
  目的是为学生展示"线性模型 vs 树模型"在此数据集上的 AUC 差异。
"""

import numpy as np
from sklearn.tree            import DecisionTreeClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling  import SMOTE
from imblearn.pipeline       import Pipeline as ImbPipeline

from config import DT_PARAM_GRID, LR_PARAM_GRID, CV_FOLDS, SCORING, RANDOM_STATE


# ── 公共 CV 配置 ───────────────────────────────────────────────────────────────

def _make_cv() -> StratifiedKFold:
    """StratifiedKFold 保证每折中正负类比例一致，防止偶然因子影响评估"""
    return StratifiedKFold(
        n_splits     = CV_FOLDS,
        shuffle      = True,
        random_state = RANDOM_STATE,
    )


# ── 决策树网格搜索 ─────────────────────────────────────────────────────────────

def run_dt_grid_search(
    X_train:      np.ndarray,
    y_train:      np.ndarray,
    strategy_name: str,
    class_weight  = None,
    use_smote_pipeline: bool = False,
) -> DecisionTreeClassifier:
    """
    对单一不平衡处理策略执行决策树 GridSearchCV。

    class_weight=None    → SMOTE/UnderSample 策略（数据已平衡或通过 Pipeline 平衡）
    class_weight='balanced' → ClassWeight 策略（让树内部补偿不平衡）

    use_smote_pipeline=True 时，将 SMOTE 放入 Pipeline 在每个 CV 折内执行，
    防止数据泄露（正确做法）。此时 X_train/y_train 为原始未重采样数据。

    评分指标: roc_auc（直接优化 AUC，不受分类阈值影响）
    """
    dt = DecisionTreeClassifier(
        random_state = RANDOM_STATE,
        class_weight = class_weight,
    )

    if use_smote_pipeline:
        # SMOTE 放入 Pipeline：每折训练集独立合成，防止测试集信息泄露
        smote = SMOTE(k_neighbors=5, random_state=RANDOM_STATE)
        pipeline = ImbPipeline([
            ('smote', smote),
            ('clf',   dt),
        ])
        # GridSearch 参数名需加前缀 'clf__'
        pipe_param_grid = {
            f'clf__{k}': v for k, v in DT_PARAM_GRID.items()
        }
        gs = GridSearchCV(
            estimator  = pipeline,
            param_grid = pipe_param_grid,
            cv         = _make_cv(),
            scoring    = SCORING,
            n_jobs     = -1,
            verbose    = 0,
            refit      = True,
        )
        gs.fit(X_train, y_train)
        best = {k.replace('clf__', ''): v for k, v in gs.best_params_.items()}
        # 提取 Pipeline 中的 DT 最终模型
        best_model = gs.best_estimator_.named_steps['clf']
    else:
        gs = GridSearchCV(
            estimator  = dt,
            param_grid = DT_PARAM_GRID,
            cv         = _make_cv(),
            scoring    = SCORING,
            n_jobs     = -1,
            verbose    = 0,
            refit      = True,
        )
        gs.fit(X_train, y_train)
        best = gs.best_params_
        best_model = gs.best_estimator_

    print(f"  [{strategy_name}]")
    print(f"    criterion={best['criterion']}, "
          f"max_depth={best['max_depth']}, "
          f"min_samples_leaf={best['min_samples_leaf']}, "
          f"min_samples_split={best['min_samples_split']}")
    print(f"    CV-AUC: {gs.best_score_:.4f}")

    return best_model


def train_all_dt_models(
    strategies:    dict,
    X_train_orig:  np.ndarray,
    y_train_orig:  np.ndarray,
) -> dict:
    """
    遍历三种不平衡处理策略，分别做网格搜索并返回最优决策树。

    SMOTE 策略使用 Pipeline（SMOTE 在 CV 每折内执行，避免数据泄露）。
    UnderSample 和 ClassWeight 策略使用预先重采样数据。

    参数：
        strategies:    imbalance.get_strategies() 的返回值
                       { 策略名: (X_train, y_train, class_weight) }
        X_train_orig:  原始（未重采样）训练集特征，供 SMOTE Pipeline 使用
        y_train_orig:  原始（未重采样）训练集标签
    返回：
        { 策略名: 已拟合的最优 DecisionTreeClassifier }
    """
    print("[决策树] GridSearchCV + 5-Fold CV (scoring=roc_auc)")
    print("-" * 55)
    fitted_models = {}
    for name, (X_tr, y_tr, cw) in strategies.items():
        if 'SMOTE' in name:
            # SMOTE 策略：使用原始数据 + Pipeline，正确处理每折数据泄露
            fitted_models[name] = run_dt_grid_search(
                X_train_orig, y_train_orig, name, cw,
                use_smote_pipeline=True,
            )
        else:
            fitted_models[name] = run_dt_grid_search(
                X_tr, y_tr, name, cw,
                use_smote_pipeline=False,
            )
    return fitted_models


# ── 逻辑回归对比模型 ───────────────────────────────────────────────────────────

def train_lr_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> LogisticRegression:
    """
    训练逻辑回归基准模型（用于 ROC 曲线对比）。

    为何选择 LR 作为对比？
      - LR 是线性分类器，决策边界为超平面
      - 决策树能捕捉非线性特征交互，理论上应优于 LR
      - 通过 ROC 对比，直观说明"树模型 vs 线性模型"的优劣

    LR 参数说明：
      C        : 正则化强度倒数（越小正则化越强）
      penalty  : l2 = Ridge 正则化，所有权重均匀压缩
      solver   : lbfgs 支持 l2 正则化，收敛速度快
      max_iter : 最大迭代次数，设置大以确保收敛
    """
    print("\n[逻辑回归（对比）] GridSearchCV + 5-Fold CV (scoring=roc_auc)")
    print("-" * 55)

    lr = LogisticRegression(
        random_state = RANDOM_STATE,
        class_weight = 'balanced',
        max_iter     = 1000,
    )

    gs = GridSearchCV(
        estimator  = lr,
        param_grid = LR_PARAM_GRID,
        cv         = _make_cv(),
        scoring    = SCORING,
        n_jobs     = -1,
        verbose    = 0,
        refit      = True,
    )
    gs.fit(X_train, y_train)

    best = gs.best_params_
    print(f"  [LR (ClassWeight)] "
          f"C={best['C']}, penalty={best['penalty']}, solver={best['solver']}")
    print(f"  CV-AUC: {gs.best_score_:.4f}")

    return gs.best_estimator_
