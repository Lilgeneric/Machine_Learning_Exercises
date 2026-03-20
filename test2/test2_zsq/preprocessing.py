"""
preprocessing.py - 数据加载、特征工程、编码、缩放

流程：
  1. load_raw()         → 读取原始 CSV，修复 TotalCharges 类型问题
  2. feature_engineer() → 手工构造高信息量特征（编码前操作）
  3. encode()           → Binary / Label Encoding / One-Hot Encoding
  4. split_scale()      → 训练/测试集划分 + StandardScaler 标准化

编码策略说明：
  - 目标变量 Churn:    Yes→1, No→0
  - gender:           Female→0, Male→1
  - 二值 Yes/No 列:    Yes→1, No→0（Partner, Dependents, PhoneService, PaperlessBilling）
  - Contract（有序）:  Month-to-month→0, One year→1, Two year→2
                       保留合约期越长客户粘性越高的物理含义
  - MultipleLines:    No phone service→0, No→1, Yes→2（有序）
  - 互联网附加服务:    No internet service→0, No→1, Yes→2（有序）
  - InternetService:  One-Hot（DSL/Fiber optic/No，无自然顺序）
  - PaymentMethod:    One-Hot（4 种支付方式，无自然顺序）
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_PATH, RANDOM_STATE, TEST_SIZE,
    CONTRACT_ORDER, MULTIPLELINES_ORDER, INTERNET_ADDON_ORDER,
    BINARY_YES_NO_COLS, NOMINAL_COLS, INTERNET_ADDON_COLS,
)


# ── Step 1: 数据加载 ───────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    """
    读取原始 CSV，修复已知数据质量问题：
      - TotalCharges 列以 object 存储，11 个空字符串需转为 NaN 再填充
      - customerID 无预测价值，直接丢弃
    """
    df = pd.read_csv(DATA_PATH)
    print(f"[数据加载] 原始形状: {df.shape}")

    # 目标分布
    vc = df['Churn'].value_counts()
    print(f"[数据加载] 流失分布:\n{vc}")
    churn_rate = (df['Churn'] == 'Yes').mean() * 100
    print(f"[数据加载] 流失率: {churn_rate:.2f}%  ← 类别不平衡")

    # 丢弃无意义标识列
    df = df.drop(columns=['customerID'])

    # 修复 TotalCharges：空字符串 → NaN → 中位数填充
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    n_missing = df['TotalCharges'].isna().sum()
    if n_missing > 0:
        median_val = df['TotalCharges'].median()
        df['TotalCharges'] = df['TotalCharges'].fillna(median_val)
        print(f"[数据加载] TotalCharges 缺失值 {n_missing} 条 → 中位数({median_val:.2f})填充")

    return df


# ── Step 2: 特征工程 ───────────────────────────────────────────────────────────

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    在编码前基于原始字符串列构造业务特征：

    1. num_services       : 客户订阅的服务总数（0～7）
                            高服务数量 → 更高粘性 → 更低流失风险
                            原始列: PhoneService + 6 个互联网附加服务
    2. has_internet       : 是否有互联网服务（1=有，0=无）
                            无网络服务的客户通常流失率更低（依赖更少）
    3. monthly_to_total   : MonthlyCharges / (TotalCharges + 1)
                            比值高 → 客户较新或短期内消费剧增 → 更易流失
    4. avg_monthly_charge : TotalCharges / (tenure + 1)
                            衡量历史平均月费，与 MonthlyCharges 差异反映费率变化
    5. is_new_customer    : tenure <= 12（0/1 二值）
                            新客户（入网不足 1 年）流失风险显著更高
    6. charge_per_service : MonthlyCharges / (num_services + 1)
                            单服务均摊月费，高价低服务 → 更容易觉得不划算
    """
    d = df.copy()

    # ── num_services（在编码前统计"Yes"字符串）────────────────────────────────
    service_cols = [
        'PhoneService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
    ]
    d['num_services'] = (d[service_cols] == 'Yes').sum(axis=1)

    # ── has_internet ──────────────────────────────────────────────────────────
    d['has_internet'] = (d['InternetService'] != 'No').astype(int)

    # ── 数值衍生特征 ──────────────────────────────────────────────────────────
    d['monthly_to_total']   = d['MonthlyCharges'] / (d['TotalCharges'] + 1)
    d['avg_monthly_charge'] = d['TotalCharges'] / (d['tenure'] + 1)
    d['is_new_customer']    = (d['tenure'] <= 12).astype(int)
    d['charge_per_service'] = d['MonthlyCharges'] / (d['num_services'] + 1)

    # ── 交互特征（高风险组合）────────────────────────────────────────────────
    # Month-to-month 合约 AND 新客户：双重流失风险叠加
    d['risky_combo'] = (
        (d['Contract'] == 'Month-to-month') & (d['tenure'] <= 12)
    ).astype(int)

    # 光纤宽带用户费用压力：Fiber optic 月费通常最高，高费用感知强
    d['fiber_high_charge'] = (
        (d['InternetService'] == 'Fiber optic') &
        (d['MonthlyCharges'] > d['MonthlyCharges'].median())
    ).astype(int)

    # 月费与同龄客户的相对水平（分位数特征，决策树可直接利用）
    d['monthly_charge_rank'] = d['MonthlyCharges'].rank(pct=True)

    # tenure 分段：新(0-12) / 成长(13-24) / 稳定(25-48) / 忠实(49+)
    d['tenure_segment'] = pd.cut(
        d['tenure'],
        bins   = [-1, 12, 24, 48, 9999],
        labels = [0, 1, 2, 3],
    ).astype(int)

    new_feats = (
        "num_services, has_internet, monthly_to_total, avg_monthly_charge, "
        "is_new_customer, charge_per_service, risky_combo, fiber_high_charge, "
        "monthly_charge_rank, tenure_segment"
    )
    print(f"[特征工程] 新增特征: {new_feats}")
    return d


# ── Step 3: 编码 ───────────────────────────────────────────────────────────────

def encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    全量编码流水线：
      1. 目标变量 Churn:    Yes→1, No→0
      2. gender:           Female→0, Male→1
      3. 二值 Yes/No:       Yes→1, No→0
      4. Contract:         有序 Label Encoding（0/1/2）
      5. MultipleLines:    有序 Label Encoding（0/1/2）
      6. 互联网附加服务:    有序 Label Encoding（0/1/2）
      7. InternetService, PaymentMethod: One-Hot Encoding
    """
    d = df.copy()

    # 目标变量
    d['Churn'] = (d['Churn'] == 'Yes').astype(int)

    # gender
    d['gender'] = (d['gender'] == 'Male').astype(int)

    # 二值 Yes/No 列
    for col in BINARY_YES_NO_COLS:
        d[col] = (d[col] == 'Yes').astype(int)

    # Contract 有序编码
    d['Contract'] = d['Contract'].map(CONTRACT_ORDER)

    # MultipleLines 有序编码
    d['MultipleLines'] = d['MultipleLines'].map(MULTIPLELINES_ORDER)

    # 互联网附加服务有序编码
    for col in INTERNET_ADDON_COLS:
        d[col] = d[col].map(INTERNET_ADDON_ORDER)

    # One-Hot Encoding（无序类别）
    d = pd.get_dummies(d, columns=NOMINAL_COLS, drop_first=False)

    # 确保所有列为数值类型
    bool_cols = d.select_dtypes(include='bool').columns
    d[bool_cols] = d[bool_cols].astype(int)

    print(f"[编码] 编码后特征总数: {d.shape[1] - 1}")
    return d


# ── Step 4: 划分 + 标准化 ──────────────────────────────────────────────────────

def split_scale(df: pd.DataFrame):
    """
    训练/测试集划分（stratified）+ StandardScaler 标准化。

    为何决策树场景下也做标准化？
      - 决策树本身对特征尺度不敏感（基于分裂阈值），不需要缩放
      - 但本项目同时训练逻辑回归作对比，LR 对特征量纲敏感
      - 统一缩放保证对比实验的公平性
      - 注意：Scaler 仅在训练集上 fit，防止测试集信息泄露

    返回：
        X_train, X_test, y_train, y_test, feature_names
    """
    X = df.drop('Churn', axis=1).astype(float)
    y = df['Churn'].astype(int)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,          # 保证训练/测试集流失比例一致
    )

    scaler      = StandardScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_test_sc   = scaler.transform(X_test)

    print(f"[划分] 训练集: {X_train_sc.shape}, 测试集: {X_test_sc.shape}")
    print(f"[划分] 训练集流失率: {y_train.mean()*100:.2f}%  "
          f"测试集流失率: {y_test.mean()*100:.2f}%")

    return X_train_sc, X_test_sc, y_train, y_test, feature_names
