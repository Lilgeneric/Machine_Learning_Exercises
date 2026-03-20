"""
imbalance.py - 类别不平衡处理策略

电信流失数据集约 26.5% 为正类（流失），存在轻度不平衡。
若直接训练，模型偏向预测"不流失"即可得到 ~73% 准确率，
但这对业务完全无价值——我们最关心的是找到"即将流失"的客户。

三种处理策略：
  1. SMOTE（过采样）     : 在少数类样本之间合成新样本，平衡正负类
                           优点：不丢弃多数类信息
                           缺点：合成样本与真实分布有偏差
  2. Random UnderSampling : 随机删除多数类样本直到比例 1:1
                           优点：快速，减少训练集规模加速训练
                           缺点：丢弃原始信息，可能造成泛化不足
  3. ClassWeight='balanced': 模型层面赋予少数类更高损失权重
                           优点：不改变训练数据分布，最稳健
                           缺点：超参数与数据量之间的权衡较难把握
"""

import numpy as np
from imblearn.over_sampling  import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from config import RANDOM_STATE


def get_strategies(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    """
    构建三种不平衡处理策略并返回：

    返回格式：
        {
          策略名: (X_resampled, y_resampled, class_weight_param)
        }

    其中 class_weight_param 为 None（策略1/2 已通过重采样处理）
    或 'balanced'（策略3 交由模型内部处理）。

    SMOTE 策略说明：
      - k_neighbors=5: 在 5 个最近邻之间插值生成合成样本
      - random_state: 保证可复现性
      - 仅对训练集执行，测试集保持原始分布（否则泄露信息）
    """
    orig_dist = np.bincount(y_train)
    print(f"  原始训练集分布: No={orig_dist[0]}, Yes={orig_dist[1]} "
          f"(正类占比 {orig_dist[1]/len(y_train)*100:.2f}%)")

    # ── 策略 1: SMOTE 过采样 ──────────────────────────────────────────────────
    smote = SMOTE(
        k_neighbors  = 5,
        random_state = RANDOM_STATE,
    )
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    dist_smote = np.bincount(y_smote)
    print(f"  [SMOTE]       重采样后: No={dist_smote[0]}, Yes={dist_smote[1]}")

    # ── 策略 2: Random UnderSampling ─────────────────────────────────────────
    rus = RandomUnderSampler(
        sampling_strategy = 1.0,     # 多数类:少数类 = 1:1
        random_state      = RANDOM_STATE,
    )
    X_under, y_under = rus.fit_resample(X_train, y_train)
    dist_under = np.bincount(y_under)
    print(f"  [UnderSample] 重采样后: No={dist_under[0]}, Yes={dist_under[1]}")

    # ── 策略 3: ClassWeight 平衡（不修改训练数据）────────────────────────────
    # 传入原始 X_train/y_train，class_weight='balanced' 由模型自身处理
    print(f"  [ClassWeight] 使用原始训练集，模型内部赋权 class_weight='balanced'")

    return {
        'DT (SMOTE)':       (X_smote, y_smote, None),
        'DT (UnderSample)': (X_under, y_under, None),
        'DT (ClassWeight)': (X_train, y_train, 'balanced'),
    }
