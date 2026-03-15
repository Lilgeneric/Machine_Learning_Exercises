"""
preprocessing.py - 数据加载、特征工程、编码、缩放

流程：
  1. load_raw()         → 读取原始 CSV
  2. feature_engineer() → 手工构造高信息量特征
  3. encode()           → Label Encoding + One-Hot Encoding
  4. split_scale()      → 训练/测试划分 + QuantileTransformer + 多项式展开
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, PolynomialFeatures

from config import (
    DATA_PATH, RANDOM_STATE, TEST_SIZE,
    EDU_ORDER, NOMINAL_COLS, POLY_FEATURES
)


def load_raw() -> pd.DataFrame:
    """读取原始数据集（分号分隔的 CSV）"""
    df = pd.read_csv(DATA_PATH, sep=';')
    print(f"[数据加载] 原始形状: {df.shape}")
    vc = df['y'].value_counts()
    print(f"[数据加载] 目标分布:\n{vc}")
    print(f"[数据加载] 正样本比例: {(df['y']=='yes').mean()*100:.2f}%  ← 严重不平衡")

    # 统计 unknown 分布
    unk = (df == 'unknown').sum()
    unk = unk[unk > 0]
    if len(unk):
        print(f"[数据加载] 含 unknown 的列（保留为独立类别）:\n{unk.to_string()}")
    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    手工特征工程（在编码前对原始列操作）：

    1. pdays_contacted  : 二值指示符，是否曾被联系过（pdays != 999）
                          → 96% 的样本 pdays=999 表示"从未联系"，将其替换为 0
    2. duration_log     : log(duration+1)，缓解通话时长的长尾分布
    3. econ_pressure    : emp.var.rate × euribor3m，宏观经济压力综合指标
                          → 捕捉就业率与利率的乘法交互效应
    4. prev_success     : poutcome=='success' 的二值特征
                          → 历史成功联系是最强的正样本预测因子之一
    """
    d = df.copy()

    # pdays 处理：999 → 0（表示未联系），同时生成二值指示符
    d['pdays_contacted'] = (d['pdays'] != 999).astype(int)
    d['pdays']           = d['pdays'].replace(999, 0)

    # 对数平滑
    d['duration_log'] = np.log1p(d['duration'])

    # 经济压力综合指标
    d['econ_pressure'] = d['emp.var.rate'] * d['euribor3m']

    # 历史成功联系（在 One-Hot 之前提取，避免被覆盖）
    d['prev_success'] = (d['poutcome'] == 'success').astype(int)

    return d


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    特征编码：
    - 目标变量 y: yes→1, no→0
    - education (有序): Label Encoding（按受教育程度排序）
    - 无序类别列: One-Hot Encoding（保留全部类别，unknown 作为独立类）
    """
    d = df.copy()
    d['y'] = (d['y'] == 'yes').astype(int)
    d['education'] = d['education'].map(EDU_ORDER)
    d = pd.get_dummies(d, columns=NOMINAL_COLS, drop_first=False)
    print(f"[编码] 编码后特征数: {d.shape[1] - 1}")
    return d


def add_polynomial_features(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    feature_names: list,
) -> tuple:
    """
    对 POLY_FEATURES 中指定的列做二次多项式展开（interaction_only=False 保留平方项）。
    其余列原样保留，最后拼接。

    理论依据：
      - 交叉项 x_i·x_j : 捕捉特征间交互，如「通话时长 × 历史成功联系」
      - 平方项 x_i²    : 捕捉单特征的曲率效应（非线性），
                          如利率的 U 形影响（极低和极高利率均抑制储蓄意愿）
    仅对 POLY_FEATURES 中的子集展开，避免全量展开导致维度爆炸。
    """
    poly_idx = [feature_names.index(f) for f in POLY_FEATURES
                if f in feature_names]
    rest_idx = [i for i in range(len(feature_names)) if i not in poly_idx]

    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_train_poly = poly.fit_transform(X_train[:, poly_idx])
    X_test_poly  = poly.transform(X_test[:, poly_idx])

    poly_feat_names = poly.get_feature_names_out(
        [feature_names[i] for i in poly_idx]
    ).tolist()
    rest_feat_names = [feature_names[i] for i in rest_idx]

    X_train_out = np.hstack([X_train[:, rest_idx], X_train_poly])
    X_test_out  = np.hstack([X_test[:, rest_idx],  X_test_poly])
    new_names   = rest_feat_names + poly_feat_names

    n_poly = X_train_out.shape[1] - len(feature_names)
    print(f"[多项式特征] 原始维度: {len(feature_names)} → "
          f"展开后维度: {X_train_out.shape[1]} "
          f"(新增 {n_poly} 个多项式项)")
    return X_train_out, X_test_out, new_names


def split_scale(df: pd.DataFrame):
    """
    训练/测试集划分 + QuantileTransformer 归一化 + 多项式特征展开。

    为什么用 QuantileTransformer 而非 StandardScaler？
      - 数值特征（如 duration、euribor3m）分布严重偏斜，StandardScaler 无法消除偏斜
      - QuantileTransformer 将每个特征映射为正态分布，使 LR 的梯度更稳定
      - 尤其有助于 L1 正则化在高维特征上的收敛

    注意：QuantileTransformer 仅在训练集上 fit，防止测试集信息泄露。

    返回：
        X_train_sc, X_test_sc, y_train, y_test, feature_names
    """
    X = df.drop('y', axis=1).astype(float)
    y = df['y'].astype(int)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # QuantileTransformer: 映射到标准正态分布
    qt = QuantileTransformer(
        output_distribution='normal',
        n_quantiles=1000,
        random_state=RANDOM_STATE
    )
    X_train_sc = qt.fit_transform(X_train)
    X_test_sc  = qt.transform(X_test)

    # 多项式特征扩展
    X_train_sc, X_test_sc, feature_names = add_polynomial_features(
        X_train_sc, X_test_sc, feature_names
    )

    print(f"[划分] 训练集: {X_train_sc.shape}, 测试集: {X_test_sc.shape}")
    print(f"[划分] 训练集正样本比例: {y_train.mean()*100:.2f}%")
    return X_train_sc, X_test_sc, y_train, y_test, feature_names
