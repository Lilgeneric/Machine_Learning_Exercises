"""
config.py - 全局配置常量

包含路径、随机种子、超参数网格等全部可调参数。
修改此文件即可统一调整实验配置，无需改动其他模块。
"""

import os

# ── 路径 ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
OUT_DIR   = os.path.join(BASE_DIR, 'output')

# ── 实验参数 ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5

# ── 编码映射 ──────────────────────────────────────────────────────────────────

# Contract 有序编码：合同期越长，客户粘性越高
CONTRACT_ORDER = {
    'Month-to-month': 0,
    'One year':       1,
    'Two year':       2,
}

# MultipleLines 有序编码：No phone service < No < Yes
MULTIPLELINES_ORDER = {
    'No phone service': 0,
    'No':               1,
    'Yes':              2,
}

# 互联网附加服务：No internet service < No < Yes（反映服务订阅程度）
INTERNET_ADDON_ORDER = {
    'No internet service': 0,
    'No':                  1,
    'Yes':                 2,
}

# 二值 Yes/No 列（直接映射 0/1）
BINARY_YES_NO_COLS = [
    'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
]

# 需要 One-Hot 编码的无序类别列
# InternetService (DSL/Fiber optic/No) 和 PaymentMethod (4 种支付方式)
# 这些列不存在自然顺序，One-Hot 防止模型产生错误的大小关系假设
NOMINAL_COLS = ['InternetService', 'PaymentMethod']

# 互联网附加服务列（使用有序编码）
INTERNET_ADDON_COLS = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
]

# ── 决策树超参数网格 ───────────────────────────────────────────────────────────
#
# 核心参数说明：
#   max_depth        : 树的最大深度
#                      深度越大 → 拟合越精细 → 过拟合风险上升
#                      None 表示不限深度（完全生长）
#   min_samples_leaf : 叶子节点最少样本数
#                      越大 → 叶子更"纯净"但泛化能力更强（剪枝效果）
#   min_samples_split: 内部节点拆分所需的最少样本数
#                      越大 → 树更保守（减少分支）
#   criterion        : 信息增益度量
#                      gini    : 基尼不纯度（计算更快）
#                      entropy : 信息增益（对不平衡数据略优）
#   ccp_alpha        : 代价复杂度剪枝参数（Cost-Complexity Pruning）
#                      通过后剪枝移除代价收益比低的子树
#                      alpha=0.0 → 不剪枝；alpha越大 → 剪枝越激进
#                      有效范围由 cost_complexity_pruning_path 动态确定
#
DT_PARAM_GRID = {
    'max_depth':         [3, 5, 7, 10, 15, 20, None],
    'min_samples_leaf':  [1, 2, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'criterion':         ['gini', 'entropy'],
    'ccp_alpha':         [0.0, 0.0005, 0.001, 0.002, 0.005],
}

# GridSearchCV 评分指标：直接优化 AUC
# 选择 roc_auc 而非 f1 的理由：
#   - roc_auc 基于概率排序，不受分类阈值影响，更能反映模型的整体判别能力
#   - 在类别不平衡场景中，roc_auc 比 accuracy 更具参考价值
SCORING = 'roc_auc'

# ── 逻辑回归超参数网格（用于 ROC 对比）────────────────────────────────────────
LR_PARAM_GRID = {
    'C':       [0.01, 0.1, 1.0, 10.0, 100.0],
    'penalty': ['l2'],
    'solver':  ['lbfgs'],
}
