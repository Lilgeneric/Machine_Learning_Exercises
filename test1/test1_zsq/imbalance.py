"""
imbalance.py - 类别不平衡处理

银行营销数据集正样本（订阅定期存款）仅占 ~11%，
若不处理，模型会偏向预测多数类（"不订阅"），导致少数类召回极低。

三种策略：
  A. SMOTE 过采样   : 合成少数类邻近样本，扩充训练集至 1:1
  B. 随机欠采样     : 随机删除多数类样本，缩减训练集至 1:1
  C. class_weight   : 模型层面给少数类更高损失权重，不改变数据分布
"""

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from config import RANDOM_STATE


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    SMOTE (Synthetic Minority Over-sampling Technique)
    原理：在少数类样本的 k 近邻之间插值生成新样本，
          比单纯复制更能引入多样性，降低过拟合风险。
    """
    # sampling_strategy=0.4: 将正样本补充至多数类的 40%（而非 1:1）
    # 实验表明 0.4 比 1.0 给出更好的 AUC，过度过采样会引入大量噪声合成样本
    smote = SMOTE(sampling_strategy=0.4, random_state=RANDOM_STATE, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[SMOTE]  重采样后: {X_res.shape}, "
          f"正样本比例: {y_res.mean()*100:.1f}%")
    return X_res, y_res


def apply_undersample(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    随机欠采样 (Random Under-Sampling)
    原理：随机丢弃多数类样本直到与少数类等量。
          训练集大幅缩小，速度快，但会损失多数类信息。
    """
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    print(f"[欠采样] 重采样后: {X_res.shape}, "
          f"正样本比例: {y_res.mean()*100:.1f}%")
    return X_res, y_res


def get_strategies(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    """
    返回三种不平衡处理策略的数据，供 modeling.py 统一调用。

    返回格式：
        {
          策略名称: (X_resampled, y_resampled, class_weight_param)
        }
    class_weight_param:
        None      → 不使用类别权重（SMOTE/欠采样已平衡数据）
        'balanced'→ 自动按类频率反比加权（class_weight 策略）
    """
    X_smote,  y_smote  = apply_smote(X_train, y_train)
    X_under,  y_under  = apply_undersample(X_train, y_train)

    return {
        'LR (SMOTE)':       (X_smote, y_smote,  None),
        'LR (UnderSample)': (X_under, y_under,  None),
        'LR (ClassWeight)': (X_train, y_train,  'balanced'),
    }
