A simple machine learning repo for share. There are some usefull tools.

about example_ml.py

方法简介

  对于离散无序变量，特征工程使用pd.dummies转为one-hot特征。

  对于连续数据，特征工程进行标准化处理；比较特征，作为离散有序数据or连续数据对待?

  对于离散有序数据，首先作为连续数据探索与因变量的关系，再根据实际情况讨论作为其他数据类型的可行性。


建模流程：

  数据探索：读取数据、缺失值、变量类型、数据类型等；

  数据清洗：变量分类、类型转换、归一化or标准化；

  特征工程：特征相关性分析、特征构造、特征选择；

  模型构建：初始模型、集成模型，调参模型，交叉验证；

  指标评价：F1-score、AUC分析、内存占用；

