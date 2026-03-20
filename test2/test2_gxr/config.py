"""
config.py — 全局配置常量
"""

import os

# ── 路径 ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(BASE_DIR, '..', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
OUT_DIR      = os.path.join(BASE_DIR, 'output')

# ── 随机种子 ─────────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── 数据集划分 ───────────────────────────────────────────────────────────────
TEST_SIZE    = 0.2

# ── 交叉验证 ─────────────────────────────────────────────────────────────────
CV_FOLDS     = 5

# ── 特征编码规则 ──────────────────────────────────────────────────────────────
# 有序编码：合约类型（粘性递增）
CONTRACT_ORDER = {
    'Month-to-month': 0,
    'One year':       1,
    'Two year':       2,
}

# 二元特征（Yes→1 / No→0）
BINARY_COLS = [
    'Partner', 'Dependents', 'PhoneService',
    'PaperlessBilling',
]

# One-Hot 编码的无序分类特征
ONEHOT_COLS = [
    'InternetService', 'PaymentMethod',
    'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies',
]

# ── 决策树超参数搜索空间 ───────────────────────────────────────────────────────
DT_PARAM_GRID = {
    'criterion':        ['gini', 'entropy'],
    'max_depth':        [3, 5, 7, 10, 15, None],
    'min_samples_leaf': [1, 5, 10, 20],
    'min_samples_split':[2, 5, 10],
    'ccp_alpha':        [0.0, 0.0001, 0.001],
}

# ── Matplotlib / Seaborn 样式 ──────────────────────────────────────────────
PLOT_STYLE   = 'whitegrid'
PLOT_PALETTE = 'muted'
DPI          = 150
