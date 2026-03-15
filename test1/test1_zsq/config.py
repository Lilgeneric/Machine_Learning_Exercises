"""
config.py - 全局配置

数据集背景：bank-additional-full（Moro et al. 2014）
  正样本（订阅定期存款）仅占 ~11.3%，严重类别不平衡。
  根据原论文及大量实验验证，逻辑回归在此数据集的 AUC 上限约为 0.95～0.96。
  本项目通过 QuantileTransformer + 二阶多项式特征工程，尽最大努力逼近该上限。
"""
import os

# ── 路径 ──────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'bank-additional-full.csv')
OUT_DIR   = os.path.join(BASE_DIR, 'output')

# ── 实验参数 ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 3          # 3-fold: 在训练时间与泛化评估之间取平衡

# ── 超参数网格 ────────────────────────────────────────────────────────────────
# 经过网格扫描实验确定最优 C 范围：特征维度高（~500+），需要较强正则化（小 C）
#
# C       : 正则化强度的倒数，越小正则化越强
#             损失 = (1/C) * ||w||^p + Σ log(1 + exp(-y_i · wᵀxᵢ))
#             C 大 → 更信任数据，权重大，可能过拟合高维特征
#             C 小 → 更依赖先验（正则化），权重趋零/稀疏，防止过拟合
#
# penalty : l1 (Lasso) → 稀疏解，相当于自动特征选择，适合高维多项式特征
#           l2 (Ridge) → 均匀压缩所有权重，适合特征间高度相关的场景
PARAM_GRID = [
    {
        'penalty' : ['l1'],
        'C'       : [0.005, 0.01, 0.02, 0.05],
        'solver'  : ['liblinear'],
        'max_iter': [3000],
    },
    {
        'penalty' : ['l2'],
        'C'       : [0.005, 0.01, 0.02, 0.05],
        'solver'  : ['lbfgs'],
        'max_iter': [3000],
    },
]

# ── 二阶多项式特征展开列表 ─────────────────────────────────────────────────────
# 包含连续型特征（通话、经济指标）和关键二值 One-Hot 特征
# interaction_only=False: 同时保留 x_i² 项（捕捉单变量曲率）
POLY_FEATURES = [
    # 通话特征（最强预测变量）
    'duration',
    'duration_log',
    # 宏观经济特征（强相关，捕捉交互意义大）
    'euribor3m',
    'nr.employed',
    'emp.var.rate',
    'cons.price.idx',
    'cons.conf.idx',
    # 衍生特征
    'pdays_contacted',
    'econ_pressure',
    'prev_success',
    'previous',
    'pdays',
    'education',
    # 关键 One-Hot 特征（与通话时长的交互极具预测力）
    'poutcome_success',
    'poutcome_nonexistent',
    'contact_cellular',
    'month_may',
    'month_nov',
    'month_oct',
    'month_mar',
    'month_sep',
    'month_apr',
    'month_jun',
    'month_jul',
    'month_aug',
    'month_dec',
    'job_student',
    'job_retired',
    'marital_single',
]

# ── 有序特征编码映射（Label Encoding） ────────────────────────────────────────
EDU_ORDER = {
    'illiterate'         : 0,
    'basic.4y'           : 1,
    'basic.6y'           : 2,
    'basic.9y'           : 3,
    'high.school'        : 4,
    'professional.course': 5,
    'university.degree'  : 6,
    'unknown'            : 3,   # 以中等水平作为缺省
}

# ── 无序类别特征（One-Hot Encoding） ─────────────────────────────────────────
NOMINAL_COLS = [
    'job', 'marital', 'default', 'housing', 'loan',
    'contact', 'month', 'day_of_week', 'poutcome',
]
