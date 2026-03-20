# 基于决策树算法的电信客户流失预测

## 实验概述

本实验使用 Kaggle Telco Customer Churn 数据集，构建决策树分类器对电信客户流失进行预测。重点涵盖特征工程、类别不平衡处理、超参数调优以及多维度性能评估。

**数据集**：[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- 样本量：7,043 条客户记录
- 特征数：20 个原始特征（含数值型和类别型）
- 目标变量：`Churn`（流失=1，未流失=0）
- 类别分布：未流失 73.5%，流失 26.5%（轻度不平衡）

---

## 项目结构

```
test2_zsq/
├── config.py          # 全局配置（路径、超参数网格、编码映射）
├── preprocessing.py   # 数据加载、特征工程、编码、标准化
├── imbalance.py       # 类别不平衡处理（SMOTE / 欠采样 / ClassWeight）
├── modeling.py        # 决策树 GridSearchCV 调优 + LR 对比模型
├── evaluation.py      # 指标计算与可视化（混淆矩阵、ROC 曲线、特征重要性）
├── main.py            # 主执行入口（完整流水线）
├── requirements.txt   # Python 依赖
└── output/            # 自动生成的图表目录
    ├── metrics_bar.png         Precision/Recall/F1 柱状图
    ├── confusion_matrices.png  混淆矩阵对比图
    ├── roc_curves.png          ROC 曲线对比图（含 LR 基准）
    └── feature_importance.png  决策树特征重要性 Top-20
```

---

## 快速运行

```bash
conda activate ml
cd test2/test2_zsq
python main.py
```

---

## 方法论

### 1. 数据预处理

**问题修复**
- `TotalCharges` 列以 `object` 类型存储，含 11 条空字符串 → 转为 `NaN` 并用中位数填充
- 删除无预测价值的 `customerID` 列

**特征工程**（在编码前构造）

| 新特征 | 计算方式 | 业务含义 |
|--------|----------|----------|
| `num_services` | 7 项服务中订阅"Yes"的数量 | 服务越多客户粘性越高 |
| `has_internet` | `InternetService != 'No'` | 有网络服务的客户流失风险更复杂 |
| `monthly_to_total` | `MonthlyCharges / (TotalCharges + 1)` | 比值高 → 客户较新 → 流失风险高 |
| `avg_monthly_charge` | `TotalCharges / (tenure + 1)` | 历史均摊月费 |
| `is_new_customer` | `tenure <= 12` | 入网不足 1 年的新客户流失率显著更高 |
| `charge_per_service` | `MonthlyCharges / (num_services + 1)` | 单服务均摊费用高 → 性价比感知低 |

**特征编码策略**

| 特征类型 | 编码方式 | 示例 |
|----------|----------|------|
| 二值 Yes/No | Label Encoding (0/1) | Partner, Dependents, PhoneService |
| 性别 | Label Encoding | Female=0, Male=1 |
| 合约类型（有序） | Label Encoding | Month-to-month=0, One year=1, Two year=2 |
| 多线路（有序） | Label Encoding | No phone service=0, No=1, Yes=2 |
| 互联网附加服务（有序） | Label Encoding | No internet service=0, No=1, Yes=2 |
| 互联网类型、支付方式（无序） | One-Hot Encoding | 避免错误的大小关系假设 |

**为什么 Contract 用有序编码而非 One-Hot？**
> 合约期越长的客户，通常绑定越深，流失风险越低。`Two year(2) > One year(1) > Month-to-month(0)` 的数值大小关系与业务逻辑一致，保留这种顺序信息有助于决策树学习到更准确的分裂规则。

### 2. 类别不平衡处理

数据集约 26.5% 为流失客户。若直接训练，模型可能只需预测"所有人都不流失"就能达到 73%+ 准确率，但这在业务上毫无价值。

| 策略 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **SMOTE 过采样** | 在少数类样本间插值合成新样本 | 保留多数类信息，增加多样性 | 合成样本可能偏离真实分布 |
| **随机欠采样** | 随机删除多数类样本至 1:1 | 训练速度快，去噪效果好 | 丢失原始信息，泛化能力受损 |
| **ClassWeight='balanced'** | 模型内部给少数类更高损失权重 | 不改变数据分布，最稳健 | 需配合合适的超参数范围 |

### 3. 超参数调优（决策树）

使用 **GridSearchCV + StratifiedKFold（k=5）** 搜索最优参数。

| 参数 | 搜索范围 | 含义 |
|------|----------|------|
| `max_depth` | [3, 5, 7, 10, 15, 20, None] | 树的最大深度，控制复杂度上限 |
| `min_samples_leaf` | [1, 2, 5, 10, 20] | 叶子节点最少样本数，等价于剪枝 |
| `min_samples_split` | [2, 5, 10] | 内部节点拆分所需最少样本数 |
| `criterion` | ['gini', 'entropy'] | 特征选择的信息度量方式 |

**搜索规模**：7 × 5 × 3 × 2 = **210 组参数** × 5 折 = 1050 次拟合/策略（`n_jobs=-1` 并行）

**评分指标**：`roc_auc`（直接优化 AUC，不受分类阈值影响，适合不平衡场景）

### 4. 性能评估

#### 为什么不能只看准确率？

```
假设模型策略：所有样本预测为"未流失"
Accuracy = 73.5%（看似不错）
Recall(流失) = 0%（实际毫无价值）
```

在客户流失预测中，**漏报（FN）的代价远高于误报（FP）**：
- 漏掉一个流失客户 → 客户离开，业务损失
- 误报一个未流失客户 → 浪费一次客服资源

因此重点关注：
- **Recall（流失类）**：所有真实流失客户中，被模型找到的比例
- **F1-Score（流失类）**：Precision 与 Recall 的调和平均，综合衡量
- **AUC**：模型对流失/不流失客户的整体排序能力

---

## 输出图表说明

### `metrics_bar.png`
分组柱状图，对比 3 种 DT 策略 + LR 在 Precision/Recall/F1/Accuracy 上的表现。
右侧水平条形图突出展示流失类(1)的 Precision-Recall-F1 权衡关系。

### `confusion_matrices.png`
四个模型的混淆矩阵并排对比：
- **左下格（FN）越小越好** → 减少流失客户的漏报
- **右下格（TP）越大越好** → 成功识别流失客户

### `roc_curves.png`
ROC 曲线对比图（含全图 + 关键区域放大）：
- 曲线越靠近左上角，AUC 越高，模型判别能力越强
- 粉红色虚线为 LR 基准，用于说明决策树 vs 线性模型的差异

### `feature_importance.png`
决策树三种策略的特征重要性（Gini Importance）Top-20 横向柱状图：
- 值越高的特征在树分裂中降低基尼不纯度的贡献越大
- 有助于理解哪些特征对流失预测最关键（如 Contract、tenure、MonthlyCharges）

---

## DT vs LR 对比思考

决策树的核心优势在于能够捕捉**非线性特征交互**：
- 例如："合约类型=Month-to-month **且** 使用时长 < 12个月 **且** MonthlyCharges > 70" 这类组合规则，DT 可以直接在叶子节点表示
- 而 LR 只能学习线性加权组合，无法天然表达这类"与"逻辑

在充分特征工程后，DT 通常能在 AUC 上超越 LR，尤其是当数据存在明显的分段/门限效应时。

若需进一步提升：可考虑 **Random Forest** 或 **XGBoost** 等集成树方法，通常能将 AUC 提升至 0.88+。

---

## 环境依赖

```bash
conda activate ml
pip install -r requirements.txt
```

| 库 | 版本要求 | 用途 |
|----|----------|------|
| pandas | ≥1.5.0 | 数据处理 |
| numpy | ≥1.23.0 | 数值计算 |
| scikit-learn | ≥1.2.0 | 模型训练、GridSearchCV、评估 |
| imbalanced-learn | ≥0.10.0 | SMOTE、RandomUnderSampler |
| matplotlib | ≥3.6.0 | 可视化 |
| seaborn | ≥0.12.0 | 样式增强 |
