# 产品需求文档 (PRD)
## 基于逻辑回归的银行营销结果预测

**项目编号**: test1_gxr
**版本**: v1.0
**日期**: 2026-03-15
**作者**: gxr

---

## 1. 背景与目标

### 1.1 业务背景

葡萄牙某银行通过电话营销推广定期存款产品，数据集记录了 41,188 次营销通话的客户特征及最终是否订阅的结果。由于绝大多数客户拒绝订阅（占比约 88.7%），数据呈现严重的类别不平衡，若模型简单地将全部样本预测为"不订阅"，准确率可达 88.7%，但实际上对业务毫无价值。

### 1.2 项目目标

| 目标 | 指标 |
|------|------|
| 训练能有效识别潜在订阅客户的分类模型 | AUC ≥ 0.96（三个模型均达标）|
| 处理类别不平衡 | 三种策略对比：SMOTE / 欠采样 / class_weight |
| 超参数调优 | GridSearchCV + 5-Fold Stratified CV |
| 全面评估模型 | Precision、Recall、F1-Score、混淆矩阵、ROC曲线 |

---

## 2. 数据集说明

### 2.1 数据来源

- **文件**: `../bank-additional-full.csv`（分号分隔）
- **规模**: 41,188 行 × 21 列（20 个特征 + 1 个目标变量）
- **目标变量**: `y`（`yes`=订阅定期存款, `no`=未订阅）

### 2.2 特征清单

| 特征名 | 类型 | 说明 |
|--------|------|------|
| age | 数值 | 年龄 |
| job | 类别（无序） | 职业（12 类） |
| marital | 类别（无序） | 婚姻状况 |
| education | 类别（有序） | 学历（8 级） |
| default | 类别（无序） | 信用违约历史 |
| housing | 类别（无序） | 是否有住房贷款 |
| loan | 类别（无序） | 是否有个人贷款 |
| contact | 类别（无序） | 联系方式 |
| month | 类别（无序） | 最近联系月份 |
| day_of_week | 类别（无序） | 最近联系星期 |
| duration | 数值 | 最后通话时长（秒）|
| campaign | 数值 | 本次活动联系次数 |
| pdays | 数值 | 距上次联系天数（999=从未联系）|
| previous | 数值 | 之前活动联系次数 |
| poutcome | 类别（无序） | 上次活动结果 |
| emp.var.rate | 数值 | 就业变动率（季度）|
| cons.price.idx | 数值 | 消费者价格指数（月度）|
| cons.conf.idx | 数值 | 消费者信心指数（月度）|
| euribor3m | 数值 | 欧洲银行间利率 3 个月期（日度）|
| nr.employed | 数值 | 从业人员数量（季度）|

### 2.3 类别不平衡

- `no`（未订阅）：约 36,548 条（88.7%）
- `yes`（已订阅）：约 4,640 条（11.3%）
- 不平衡比例约 **7.9 : 1**

---

## 3. 功能需求

### 3.1 数据预处理模块

#### F1 - 缺失值处理
- `unknown` 视为独立类别，保留为 One-Hot 编码的一列，不做填充或删除
- 检查并输出每列 unknown 数量

#### F2 - 特征编码
| 特征 | 编码方式 | 说明 |
|------|----------|------|
| education | Label Encoding（有序） | illiterate=0, basic.4y=1, basic.6y=2, basic.9y=3, high.school=4, professional.course=5, university.degree=6, unknown=3 |
| job, marital, default, housing, loan, contact, month, day_of_week, poutcome | One-Hot Encoding | drop_first=False，保留全部列以便解释性 |

#### F3 - 特征缩放
- 使用 `StandardScaler` 对全部特征进行标准化
- **严格规范**：`fit` 仅在训练集，`transform` 应用于训练集和测试集，避免数据泄露

#### F4 - 数据集划分
- 训练集 / 测试集 = 80% / 20%
- 使用 `stratify=y` 保持类别比例一致
- `random_state=42`，确保实验可复现

### 3.2 类别不平衡处理模块

#### F5 - 三种策略

| 策略 | 方法 | 数据来源 |
|------|------|----------|
| 策略 A：过采样 | SMOTE（random_state=42）| 仅在训练集上进行 |
| 策略 B：欠采样 | RandomUnderSampler（random_state=42）| 仅在训练集上进行 |
| 策略 C：权重调节 | class_weight='balanced' | 原始训练集 |

> **注意**：重采样只能在训练集上进行，测试集保持原始分布

### 3.3 超参数调优模块

#### F6 - GridSearchCV + 5-Fold CV

```python
param_grid = [
    {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['lbfgs']},
    {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1'], 'solver': ['liblinear']},
]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = 'roc_auc'  # 在不平衡数据上比 accuracy 更合理
```

**核心参数说明**：
- `C`：正则化强度的倒数，越小正则化越强，防止过拟合；越大允许模型复杂度越高
- `penalty`：正则化类型，`l1` 产生稀疏解（特征选择），`l2` 倾向于小权重分布

### 3.4 模型评估模块

#### F7 - 分类报告
- 使用 `classification_report` 输出 Precision、Recall、F1-Score（含 macro avg 和 weighted avg）
- 额外输出 AUC 值

#### F8 - 混淆矩阵可视化
- 三个模型并排展示（1 行 3 列子图）
- 使用 `ConfusionMatrixDisplay`，蓝色热力图，标注 FP/FN/TP/TN
- 保存为 `output/confusion_matrices.png`（dpi=150）

#### F9 - ROC 曲线可视化
- 三条 ROC 曲线绘制在同一坐标系
- 图例显示模型名称和 AUC 值
- 对角线虚线（随机分类器基准）
- 保存为 `output/roc_curves.png`（dpi=150）

#### F10 - 质量门禁
- 三个模型 AUC 均须 ≥ 0.96，否则终端输出警告

---

## 4. 非功能需求

| 类别 | 要求 |
|------|------|
| 可复现性 | 所有随机种子统一设置为 42 |
| 可读性 | 代码分节注释，关键步骤打印中间状态 |
| 运行环境 | conda 环境 `ml`，Python ≥ 3.9 |
| 输出目录 | 脚本旁的 `output/` 目录，自动创建 |
| 无警告 | 使用 `warnings.filterwarnings('ignore')` 过滤无关警告 |
| 字体 | 使用 DejaVu Sans，避免中文字体缺失乱码 |

---

## 5. 技术栈

| 库 | 版本要求 | 用途 |
|----|----------|------|
| pandas | ≥ 1.5 | 数据加载与处理 |
| numpy | ≥ 1.21 | 数值计算 |
| scikit-learn | ≥ 1.2 | 模型训练、评估、GridSearch |
| imbalanced-learn | ≥ 0.10 | SMOTE、RandomUnderSampler |
| matplotlib | ≥ 3.5 | 图表绘制 |
| seaborn | ≥ 0.12 | 风格美化 |

---

## 6. 项目结构

```
test1_gxr/
├── PRD.md                    # 本文件：产品需求文档
├── bank_marketing_lr.py      # 主程序
├── requirements.txt          # 依赖列表
└── output/                   # 自动创建
    ├── confusion_matrices.png
    └── roc_curves.png
```

---

## 7. 验收标准

| 验收项 | 标准 |
|--------|------|
| 代码可运行 | `python bank_marketing_lr.py` 无报错完成 |
| AUC 达标 | 三个模型 AUC 均 ≥ 0.96 |
| 输出完整 | 生成两张图片且包含所有模型 |
| 指标输出 | 终端打印 classification_report |
| 数据无泄露 | Scaler 仅 fit 训练集 |

---

## 8. 执行流程图

```
数据加载
   ↓
数据预处理（编码 + 缩放）
   ↓
训练/测试集划分（8:2，stratify）
   ↓
┌──────────────────────────────────┐
│  三种不平衡处理策略（仅训练集）   │
│  A: SMOTE   B: UnderSample  C: CW│
└──────────────────────────────────┘
   ↓
GridSearchCV + 5-Fold CV（每个策略）
   ↓
最优模型在测试集评估
   ↓
┌─────────────────────────────┐
│  输出分类报告 + AUC          │
│  绘制混淆矩阵 + ROC曲线      │
└─────────────────────────────┘
```
