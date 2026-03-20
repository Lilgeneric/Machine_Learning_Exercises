"""
preprocessing.py — 数据加载、清洗、特征工程、标准化

流程：
  1. 读取原始 CSV
  2. 清洗 TotalCharges（空字符串 → NaN → 用 MonthlyCharges 填充）
  3. 标签编码目标变量 Churn (Yes→1 / No→0)
  4. 性别二元编码 (Male→1 / Female→0)
  5. 二元特征 Yes/No → 1/0
  6. 有序编码 Contract（月→0 < 年→1 < 两年→2）
  7. One-Hot 编码无序分类特征
  8. 丢弃 customerID（无信息 ID 列）
  9. 训练/测试集划分（stratified）
 10. StandardScaler 标准化（fit 仅训练集，防数据泄露）
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_PATH, RANDOM_STATE, TEST_SIZE,
    CONTRACT_ORDER, BINARY_COLS, ONEHOT_COLS,
)


def load_and_preprocess():
    """
    返回
    ----
    X_train, X_test : np.ndarray  (标准化后)
    y_train, y_test : np.ndarray
    feature_names   : list[str]
    scaler          : StandardScaler  (供后续逆变换或报告使用)
    """
    print("=" * 65)
    print("STEP 1  数据加载与概览")
    print("=" * 65)

    df = pd.read_csv(DATA_PATH)
    print(f"数据集形状   : {df.shape}")
    print(f"目标变量分布 :\n{df['Churn'].value_counts()}")
    pos_ratio = (df['Churn'] == 'Yes').mean() * 100
    print(f"正样本(Churn=Yes)比例 : {pos_ratio:.2f}%  ← 存在类别不平衡")

    # ── Step 2  数据清洗 ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 2  数据清洗 + 特征工程")
    print("=" * 65)

    df = df.copy()

    # TotalCharges 为字符串，且新客户（tenure=0）存在空值
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce')
    n_missing = df['TotalCharges'].isna().sum()
    if n_missing:
        # tenure=0 时 TotalCharges 应等于 MonthlyCharges
        df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
        print(f"TotalCharges 缺失填充 : {n_missing} 行（用 MonthlyCharges 填充）")

    # 目标变量
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    # 性别
    df['gender'] = (df['gender'] == 'Male').astype(int)

    # 二元 Yes/No 特征
    for col in BINARY_COLS:
        df[col] = (df[col] == 'Yes').astype(int)

    # 有序编码：Contract
    df['Contract'] = df['Contract'].map(CONTRACT_ORDER)

    # One-Hot 编码
    df = pd.get_dummies(df, columns=ONEHOT_COLS, drop_first=False)

    # 丢弃 ID
    df = df.drop(columns=['customerID'])

    print(f"处理后特征总数 : {df.shape[1] - 1}")
    print(f"正样本比例     : {df['Churn'].mean()*100:.2f}%")

    # ── Step 3  数据集划分 + 标准化 ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("STEP 3  数据集划分 + 标准化")
    print("=" * 65)

    X = df.drop(columns=['Churn']).astype(float)
    y = df['Churn'].astype(int)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"训练集 : {X_train.shape}   正样本: {y_train.mean()*100:.2f}%")
    print(f"测试集 : {X_test.shape}    正样本: {y_test.mean()*100:.2f}%")

    # 标准化（虽然决策树不依赖量纲，但便于与 LR 统一对比）
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)   # fit 仅训练集，防数据泄露
    X_test_sc  = scaler.transform(X_test)

    return (
        X_train_sc, X_test_sc,
        y_train.values, y_test.values,
        feature_names, scaler,
    )
