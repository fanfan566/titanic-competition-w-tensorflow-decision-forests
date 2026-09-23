# 泰坦尼克号生存预测：TensorFlow 决策森林

基于 Kaggle Titanic 竞赛数据，使用 TensorFlow Decision Forests（TF-DF）训练梯度提升树模型，预测乘客是否生还，并生成竞赛提交文件。本项目目前以 `titanic-competition-w-tensorflow-decision-forests.ipynb` 为可运行主体；下文分别记录 Notebook 已实现的流程和进一步完善的方向。

## 数据与任务

训练集为 `train.csv`，其中 `Survived` 是二分类标签（1 表示生还，0 表示未生还）；测试集为 `test.csv`，没有该标签。Notebook 在 Kaggle 环境中从 `/kaggle/input/titanic/` 读取这两个文件。预测时先获得每位乘客的生还概率，再以 0.5 为阈值转换成提交所需的 0/1 标签。

## 已实现的流程

1. **预处理**：清理 `Name` 中的部分标点并按词切分；从 `Ticket` 中提取末尾编号 `Ticket_number` 和前缀 `Ticket_item`。模型特征包含 `Pclass`、`Name`、`Sex`、`Age`、`SibSp`、`Parch`、`Fare`、`Cabin`、`Embarked` 以及两项票号衍生特征；`PassengerId`、原始 `Ticket` 和目标列 `Survived` 不进入特征列表。
2. **建模**：使用 TF-DF 的 `GradientBoostedTreesModel` 训练默认参数模型和手动设置参数的模型。手动参数示例包含 `num_trees=2000`、`shrinkage=0.05` 及稀疏斜向切分。
3. **调参**：使用 `tfdf.tuner.RandomSearch(num_trials=1000)` 搜索叶节点最小样本数、类别特征算法、树生长策略、局部树深度或全局节点数、学习率、候选特征比例及分裂轴配置。
4. **集成**：以不同随机种子训练 100 个梯度提升树模型，对预测概率求均值后生成另一份提交结果。按顺序运行整个 Notebook 时，后一次导出会覆盖前一次的 `submission.csv`。
5. **评估与输出**：通过 TF-DF 模型检查器打印内部评估的 Accuracy 和 Loss，并使用 `model.summary()` 查看模型信息；导出 `/kaggle/working/submission.csv`，列为 `PassengerId,Survived`。

TF-DF 在此流程中处理数值、类别及分词后的姓名特征；Notebook 没有显式实现数值缺失值填充和独热编码。因此，不能将这两项写成已经完成的预处理步骤。

## 运行方法

1. 在 Kaggle 创建或打开 Notebook，添加 Titanic 竞赛数据，确认存在 `/kaggle/input/titanic/train.csv` 和 `/kaggle/input/titanic/test.csv`。
2. 确认运行环境可导入 `tensorflow` 和 `tensorflow_decision_forests`；两者需要兼容的版本组合。Notebook 保存的运行输出显示当时使用的是 TF-DF 1.2.0，这不代表其他版本组合已经验证。
3. 上传并打开 `titanic-competition-w-tensorflow-decision-forests.ipynb`，从上到下运行单元格。调参的 1000 次试验以及 100 模型集成可能需要较长时间，可以先运行至第一次生成 `submission.csv` 的单元格，以验证数据和环境。
4. 从 `/kaggle/working/submission.csv` 下载结果并提交至 Titanic 竞赛。该 CSV 是离散标签；若需要逐人展示生还概率，应另外导出预测数组，而不是将现有提交列解释为概率。

在本地运行时，应将 Notebook 的 Kaggle 绝对路径改为本地 `train.csv`、`test.csv` 的实际路径，并修改提交文件的保存位置。

## 目标功能与当前状态

| 功能 | 当前 Notebook 状态 | 后续实现建议 |
| --- | --- | --- |
| 姓名、票号特征处理 | 已实现姓名分词及票号拆分 | 增加头衔提取，并处理少见头衔 |
| 缺失值与类别处理 | 未显式填补或编码 | 在训练集上拟合填补规则，保持训练集和测试集一致；为线性基线进行类别编码 |
| 家庭规模 | 未实现 | 新增 `FamilySize = SibSp + Parch + 1` |
| TF-DF 模型与调参 | 已实现 GBT、随机搜索及多种子集成 | 使用独立验证方案选择模型，记录最佳参数 |
| 逻辑回归与随机森林基线 | 未实现独立对照实验 | 在相同数据划分和指标下训练比较 |
| 特征重要性 | 可通过模型检查器扩展，当前未形成专题图表 | 输出重要性表与条形图，说明采用的重要性定义 |
| 混淆矩阵和可视化报告 | 未实现 | 在有真实标签的验证集上绘制混淆矩阵和模型对比图 |
