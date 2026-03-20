"""
imbalance.py — 类别不平衡处理策略

电信客户流失数据集中：
  Not Churn ≈ 73.5%   Churn ≈ 26.5%
若直接训练，模型倾向于预测"不流失"而获得高准确率，但漏掉大量真实流失客户。

三种策略：
  A. SMOTE       — 合成少数类样本（过采样），保留原始多数类信息
  B. UnderSample — 随机删除多数类样本（欠采样），训练集缩小但更平衡
  C. ClassWeight — 在模型初始化时设置 class_weight='balanced'，
                   给少数类更高的损失权重，无需修改数据
"""

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from config import RANDOM_STATE


def apply_imbalance_strategies(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    对训练集应用三种不平衡处理策略。

    参数
    ----
    X_train : 标准化后的训练特征
    y_train : 训练标签

    返回
    ----
    dict  {strategy_name: (X_resampled, y_resampled, class_weight_param)}
        class_weight_param 为传入 DecisionTreeClassifier 的 class_weight 参数。
        SMOTE / UnderSample 策略中模型不需要额外权重，故为 None。
    """
    print("=" * 65)
    print("STEP 4  类别不平衡处理（三种策略）")
    print("=" * 65)
    print(f"原始训练集 : {X_train.shape}  正样本比例: {y_train.mean()*100:.2f}%\n")

    strategies = {}

    # ── A. SMOTE 过采样 ──────────────────────────────────────────────────────
    # sampling_strategy=0.5 → 合成后少数类为多数类的 50%
    smote = SMOTE(sampling_strategy=0.5, random_state=RANDOM_STATE, k_neighbors=5)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    strategies['DT + SMOTE'] = (X_sm, y_sm, None)
    print(f"[A SMOTE]       {X_sm.shape}  正样本: {y_sm.mean()*100:.1f}%")

    # ── B. RandomUnderSampler 欠采样 ──────────────────────────────────────────
    rus = RandomUnderSampler(sampling_strategy=0.5, random_state=RANDOM_STATE)
    X_ru, y_ru = rus.fit_resample(X_train, y_train)
    strategies['DT + UnderSample'] = (X_ru, y_ru, None)
    print(f"[B UnderSample] {X_ru.shape}  正样本: {y_ru.mean()*100:.1f}%")

    # ── C. class_weight='balanced' ────────────────────────────────────────────
    # 数据不变，通过模型权重补偿不平衡
    strategies['DT + ClassWeight'] = (X_train, y_train, 'balanced')
    print(f"[C ClassWeight] {X_train.shape}  正样本: {y_train.mean()*100:.2f}%"
          f"  (class_weight='balanced')")

    return strategies
