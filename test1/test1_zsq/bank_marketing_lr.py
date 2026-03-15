"""
基于逻辑回归的银行营销结果预测
Bank Marketing Prediction using Logistic Regression
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import os

# ─── 0. 配置 ──────────────────────────────────────────────────────────────────
DATA_PATH = '../bank-additional-full.csv'
OUT_DIR   = 'output'
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ─── 1. 数据加载 ───────────────────────────────────────────────────────────────
print("=" * 60)
print("1. 数据加载")
print("=" * 60)

df = pd.read_csv(DATA_PATH, sep=';')
print(f"数据集形状: {df.shape}")
print(f"\n目标变量分布:\n{df['y'].value_counts()}")
print(f"\n正样本比例: {(df['y']=='yes').mean()*100:.2f}%")

# ─── 2. 数据预处理 ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. 数据预处理")
print("=" * 60)

df_proc = df.copy()

# 目标变量编码
df_proc['y'] = (df_proc['y'] == 'yes').astype(int)

# 有序特征（学历）→ Label Encoding
edu_order = {
    'illiterate': 0,
    'basic.4y': 1,
    'basic.6y': 2,
    'basic.9y': 3,
    'high.school': 4,
    'professional.course': 5,
    'university.degree': 6,
    'unknown': 3   # 视作中等水平
}
df_proc['education'] = df_proc['education'].map(edu_order)

# 无序类别特征 → One-Hot Encoding
nominal_cols = ['job', 'marital', 'default', 'housing', 'loan',
                'contact', 'month', 'day_of_week', 'poutcome']
df_proc = pd.get_dummies(df_proc, columns=nominal_cols, drop_first=False)

print(f"预处理后特征数量: {df_proc.shape[1] - 1}")

# ─── 3. 特征 / 标签分离 & 训练测试划分 ─────────────────────────────────────────
X = df_proc.drop('y', axis=1).astype(float)
y = df_proc['y'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n训练集: {X_train.shape}, 测试集: {X_test.shape}")
print(f"训练集正样本比例: {y_train.mean()*100:.2f}%")

# ─── 4. 特征缩放 ───────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─── 5. 处理类别不平衡（三种策略） ────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. 处理类别不平衡（三种策略）")
print("=" * 60)

# 策略 A: SMOTE 过采样
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_sc, y_train)
print(f"SMOTE后训练集: {X_train_smote.shape}, 正样本: {y_train_smote.mean()*100:.1f}%")

# 策略 B: 随机欠采样
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train_sc, y_train)
print(f"欠采样后训练集: {X_train_rus.shape}, 正样本: {y_train_rus.mean()*100:.1f}%")

# 策略 C: class_weight='balanced'（直接用原始数据）
X_train_cw = X_train_sc
y_train_cw = y_train

# ─── 6. 超参数调优（Grid Search + 5折交叉验证） ────────────────────────────────
print("\n" + "=" * 60)
print("4. 超参数调优 (Grid Search + 5-Fold CV)")
print("=" * 60)

# 参数空间
# C: 正则化强度的倒数，越小正则化越强
# penalty: 正则化类型 (l1 → 稀疏, l2 → 常规)
param_grid_l2 = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs'],
    'penalty': ['l2'],
    'max_iter': [1000]
}
param_grid_l1 = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear'],
    'penalty': ['l1'],
    'max_iter': [1000]
}
param_grid = [param_grid_l2, param_grid_l1]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def run_grid_search(X_tr, y_tr, label):
    gs = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    gs.fit(X_tr, y_tr)
    print(f"\n[{label}] 最佳参数: {gs.best_params_}  最佳CV-F1: {gs.best_score_:.4f}")
    return gs.best_estimator_

model_smote = run_grid_search(X_train_smote, y_train_smote, "SMOTE")
model_rus   = run_grid_search(X_train_rus,   y_train_rus,   "欠采样")
model_cw    = run_grid_search(X_train_cw,    y_train_cw,
                               "class_weight='balanced'")
# class_weight 由 best_estimator_ 从 param_grid 里来，不含 balanced，
# 所以对 CW 策略单独设置 balanced 权重
model_cw_balanced = LogisticRegression(
    **{k: v for k, v in model_cw.get_params().items()
       if k in ['C', 'penalty', 'solver', 'max_iter']},
    class_weight='balanced',
    random_state=42
)
model_cw_balanced.fit(X_train_cw, y_train_cw)
model_cw = model_cw_balanced

models = {
    'LR (SMOTE)':      (model_smote, X_train_smote, y_train_smote),
    'LR (UnderSample)':(model_rus,   X_train_rus,   y_train_rus),
    'LR (ClassWeight)':(model_cw,    X_train_cw,    y_train_cw),
}

# ─── 7. 模型评估 ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. 模型评估")
print("=" * 60)

results = {}
for name, (model, _, _) in models.items():
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc,
        'cm': confusion_matrix(y_test, y_pred)
    }
    print(f"\n{'='*40}")
    print(f"模型: {name}  AUC = {roc_auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['未订阅(0)', '订阅(1)']))

# ─── 8. 可视化：混淆矩阵 ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. 可视化 - 混淆矩阵")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Confusion Matrix Comparison', fontsize=16, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=res['cm'],
        display_labels=['No (0)', 'Yes (1)']
    )
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"{name}\nAUC = {res['auc']:.4f}", fontsize=11)

plt.tight_layout()
cm_path = os.path.join(OUT_DIR, 'confusion_matrices.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
print(f"混淆矩阵已保存: {cm_path}")
plt.close()

# ─── 9. 可视化：ROC 曲线 ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. 可视化 - ROC 曲线")
print("=" * 60)

fig, ax = plt.subplots(figsize=(8, 6))
colors = ['steelblue', 'darkorange', 'seagreen']

for (name, res), color in zip(results.items(), colors):
    ax.plot(res['fpr'], res['tpr'],
            color=color, lw=2,
            label=f"{name} (AUC = {res['auc']:.4f})")

ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Logistic Regression (3 Strategies)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

roc_path = os.path.join(OUT_DIR, 'roc_curves.png')
plt.savefig(roc_path, dpi=150, bbox_inches='tight')
print(f"ROC曲线已保存: {roc_path}")
plt.close()

# ─── 10. 汇总 ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. 结果汇总")
print("=" * 60)
print(f"{'模型':<25} {'AUC':>8}")
print("-" * 35)
for name, res in results.items():
    auc_val = res['auc']
    flag = "✓ >= 0.94" if auc_val >= 0.94 else "✗ < 0.94"
    print(f"{name:<25} {auc_val:>8.4f}  {flag}")

print("\n可视化文件:")
print(f"  混淆矩阵: {cm_path}")
print(f"  ROC曲线:  {roc_path}")
print("\n实验完成！")
