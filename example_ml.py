# coding: utf-8

# # 一、读取数据

# 读取数据，数据去重，检查缺失值

# In[112]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# read data
# train_path = "path/to/train.csv"  # with label
# test_path = "path/to/test.csv"

def read_data(path):
    data = pd.read_csv(path, index_col=False)
    data.info()
    print("------------------------------")
    
    # data duplicate
    print('After, duplicated---', len(data.duplicated()))
    # check na by col
    cols = data.columns.tolist()
    for col in cols:
        assert len(data[col].dropna()) == len(data[col])
    print('cols---\n', cols)
    return data

data = read_data(train_path)
# display(data.head())

# # 二、离散变量计数统计

# In[96]:

# your discrete and continuous features columns, need pick up by yourself.
cols_disc = ['disc_feature1', 'disc_feature2', 'disc_feature3']
cols_cont = ['cont_feature1', 'cont_feature2', 'cont_feature3']
col_y = ['label']

def fun_vc(data, col):
    '''get value_counts'''
    return data[col].value_counts().reset_index().rename(columns={'index':col,col:'count_%s'%col})    

dict_vc = {col:fun_vc(data, col) for col in cols_disc}
for col, vc_col in dict_vc.items():
    print(col, vc_col.shape)

# # 三、变量相关性分析

# In[97]:

import seaborn as sns

data_corr = data.corr()
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 6; fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size

# print('Continuous variables')
# sns.heatmap(data[cols_cont].corr(), square=True, cmap='coolwarm')
# plt.show()
# print('Discrete variable variables')
# sns.heatmap(data[cols_disc].corr(), square=True, cmap='coolwarm')
# plt.show()

print('All variables in train')
sns.heatmap(data_corr, square=True, cmap='coolwarm')
plt.show()

print("label Accept's corr in train\n", data_corr['Accept'])
print()
print("feature Price's corr in train\n", data_corr['Price'])

# Talk about corr between features and label
# Talk about corr between feature1 and feature2

# In[98]:

data[col_y].plot(kind='hist')
plt.show()

# Talk about data imbalance and how to do. Data or Metrics? 

# # 四、构造特征

# 1.离散特征转为one-hot

# In[99]:


data[cols_disc] = data[cols_disc].astype(str)
y = data['Accept']   
data = data.drop(['ID', 'Accept'], axis=1)
data_dummies = pd.get_dummies(data)
print('Before add one-hot features from disc, features are\n', data.columns.tolist())
print()
print('After add one-hot features from disc, features are\n', data_dummies.columns.tolist())


# ２．连续特征标准化后绘制箱线图

# In[100]:


X_cont = data_dummies[cols_cont]
# X_cont.boxplot()
# plt.show()
X_normalized = X_cont.apply(lambda x: (x-x.mean())/x.std())
X_normalized.loc[:, 'Date':'Price'].boxplot()
plt.show()


# ３．连续特征中删除异常数据，采用3西格玛准则

# In[101]:


X_normalized = X_normalized.reset_index()
cumsum=0
X_out = pd.DataFrame()
for col in cols_cont:
    x = X_normalized[col]
    zscore = x - x.mean()
    X_normalized[col+'_isnorm'] = zscore.abs() > 3.0 * x.std()
    X_out = X_out.append(X_normalized[X_normalized[col+'_isnorm'] == True])
    cumsum += len(X_out)
    print('exception number of %s---'%col,  len(X_out))
    #X_normalized = X_normalized[X_normalized[col+'_isnorm'] == False]

X = data_dummies.copy()
X = X.reset_index()
X = X[X['index'].isin(X_out['index'].values.tolist())==False]
X = X.drop(['index'], axis=1)  


# 4.构造交互特征和多项式特征

# In[102]:


y = y.reset_index()
y = y[y['index'].isin(X_out['index'].values.tolist())==False]
y = y.drop(['index'], axis=1)

cols_X = X.columns.tolist()
print('Before add cross and poly features:', len(cols_X))
from sklearn.preprocessing import PolynomialFeatures
#参数degree=N表示N次方项和N次交互项
poly = PolynomialFeatures(degree=3, include_bias=False)
X_cont = X.iloc[:, 0:4]
poly.fit(X_cont)
X_poly = poly.transform(X_cont) 
X = np.hstack((X_poly, X.iloc[:, 4:]))#与离散特征合并

X = pd.DataFrame(X)
X = X.apply(lambda x: (x-x.min())/(x.max()-x.min()))
cols_X = X.columns.tolist()
print('After add cross and poly features:', len(cols_X))


# # 五、搭建模型

# 1.初始模型

# In[103]:

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[106]:

import warnings
warnings.filterwarnings("ignore")
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

from sklearn.metrics import f1_score
y_pred = logreg.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred,
                            target_names=["0", "1"]))

# Talk about results0

# 2.特征筛选

# In[28]:


from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectPercentile
select = SelectPercentile(percentile=50)

#from sklearn.feature_selection import SelectFromModel
#select = SelectFromModel(
#    RandomForestClassifier(n_estimators=100, random_state=42),
#    threshold="median")

#from sklearn.feature_selection import RFE
#select = RFE(RandomForestClassifier(n_estimators=100, random_state=42),
#             n_features_to_select=50)


# In[29]:


select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)
X_test_selected = select.transform(X_test)


# 筛选后的特征可视化，黑色方块代表使用的特征，白色则未被使用。

# In[30]:


mask = select.get_support() 
# visualize the mask. black is True, white is False
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")
plt.yticks(())


# 筛选后重新预测：

# In[55]:


logreg = LogisticRegression()
logreg.fit(X_train_selected, y_train)
y_pred = logreg.predict(X_test_selected)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred,
                            target_names=["0", "1"]))

# Talk about results1 after filtered comparing with results0

# 3.AUC曲线

# In[57]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

plt.figure()

for gamma in [1]:#, 0.05, 0.01]:
    logreg.fit(X_train, y_train)
    accuracy = logreg.score(X_test, y_test)
    auc = roc_auc_score(y_test, logreg.decision_function(X_test))
    fpr, tpr, _ = roc_curve(y_test , logreg.decision_function(X_test))
    print("gamma = {:.2f}  accuracy = {:.2f}  AUC = {:.2f}".format(
        gamma, accuracy, auc))
    plt.plot(fpr, tpr, label="gamma={:.3f}".format(gamma))
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(-0.01, 1)
    plt.ylim(0, 1.02)
    plt.legend(loc="best")
    plt.show()

# AUC before and after fitered

# 4.交叉验证

# 交叉验证是一种评估泛化性能的统计学方法，它比单次划分训练集和测试集的方法更加稳定、全面。
    
# In[58]:

c, r = y_train.shape
y_train = y_train.values.reshape(c,)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(logreg, X_train, y_train, cv=5)
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))


# 5.参数调节

# (1)正则

# In[73]:


for penalty in ['l1', 'l2']:
    logres = LogisticRegression(penalty=penalty, 
              dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
              intercept_scaling=1, class_weight=None, 
              random_state=None, solver='liblinear', max_iter=1000, 
              multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)

    logreg.fit(X_train, y_train)
    print("With penalty %s"%penalty, "f1_score of : {:.4f}".format(
        f1_score(y_test, logreg.predict(X_test))))


# (2)class_weight

# In[69]:


for cw in ['None', 'balanced', {0:0.3,1:0.7}, {0:0.7,1:0.3}]:
    logres = LogisticRegression(penalty='l2', 
              dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
              intercept_scaling=1, class_weight=cw, 
              random_state=None, solver='liblinear', max_iter=1000, 
              multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)

    logreg.fit(X_train, y_train)
    print("With class_weight %s"%cw, "f1_score of : {:.4f}".format(
        f1_score(y_test, logreg.predict(X_test))))


# (3)solver

# In[70]:


for solver in ['newton-cg', 'lbfgs', 'liblinear','sag', 'saga']:
    logres = LogisticRegression(penalty='l2', 
          dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
          intercept_scaling=1, class_weight=None, 
          random_state=None, solver=solver, max_iter=1000, 
          multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)

    logreg.fit(X_train, y_train)
    print("With solver %s"%solver, "f1_score of : {:.4f}".format(
        f1_score(y_test, logreg.predict(X_test))))


# (4)multi_class

# In[72]:


for mc in ['ovr', 'multinomial']:
    logres = LogisticRegression(penalty='l2', 
      dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
      intercept_scaling=1, class_weight=None, 
      random_state=None, solver=solver, max_iter=1000, 
      multi_class=mc, verbose=0, warm_start=False, n_jobs=1)

    logreg.fit(X_train, y_train)
    print("With multi_class %s"%mc, "f1_score of : {:.4f}".format(
        f1_score(y_test, logreg.predict(X_test))))


# (5)models--随机森林

# In[82]:


from sklearn.ensemble import RandomForestClassifier
rfreg = RandomForestClassifier(n_estimators = 100, 
                               oob_score = True, 
                               n_jobs = -1,
                               random_state =50,
                               max_features = "auto", 
                               min_samples_leaf = 50)
 
rfreg.fit(X_train, y_train)
print("With RandomForestClassifier， f1_score of : {:.4f}".format(
    f1_score(y_test, rfreg.predict(X_test))))


# 随机森林作为一种基于决策树的集成算法，相当于多个决策树结果的折中。

# (6)将Beds作为连续特征

# 

# # 六、预测结果

# When use model to predict, use train+val data for training, not only train data. 

# In[116]:

data_test = read_data(test_path)

print(len(data))
print(data.columns.tolist())
print(data_test.columns.tolist())

# 1.使用全部训练数据

# In[120]:


ids = data_test["ID"].values.tolist()
X_train, y_train = X, y
display(X_train.head())
display(y_train.head())


# 2.将建模过程写为函数方便调用

# 出于预测，test data暂时没有进行异常值剔除。

# In[145]:


def build_data(data):
    cols_disc = ['disc_feature1', 'disc_feature2', 'disc_feature3']
    cols_cont = ['cont_feature1', 'cont_feature2', 'cont_feature3']
    data[cols_disc] = data[cols_disc].astype(str)
    data = data.drop(['ID'], axis=1)
    data_dummies = pd.get_dummies(data)
    X_cont = data_dummies[cols_cont]
    data_dummies[cols_cont] = data_dummies[cols_cont].apply(lambda x: (x-x.mean())/x.std())
    X = data_dummies.copy()
    print('len X:', len(X))

    cols_X = X.columns.tolist()
    print('Before add cross and poly features:', len(cols_X))

    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_cont = X.iloc[:, 0:4]
    poly.fit(X_cont)
    X_poly = poly.transform(X_cont) 
    X = np.hstack((X_poly, X.iloc[:, 4:]))#与离散特征合并
    X = pd.DataFrame(X)
    X = X.apply(lambda x: (x-x.min())/(x.max()-x.min()))
    cols_X = X.columns.tolist()
    print('After add cross and poly features:', len(cols_X))
    y = pd.DataFrame({"accept":[0 for i in range(len(X))]})
    print('len y:', len(y))
    return X, y


# 3.增加了修改数据类型的模块，解决内存占用问题

# In[167]:


X_test, y_test = build_data(data_test)

def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)
print('Before:', mem_usage(X_test))
X_train = X_train.astype(np.float16)
X_test = X_test.astype(np.float16)
print('After :', mem_usage(X_test))


# In[168]:


print('len X_train, y_train', len(X_train), len(y_train))
from sklearn.linear_model import LogisticRegression
logres = LogisticRegression(penalty=penalty, 
          dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
          intercept_scaling=1, class_weight=None, 
          random_state=None, solver='liblinear', max_iter=1000, 
              multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
logreg.fit(X_train, y_train)

from sklearn.metrics import f1_score
y_pred = logreg.predict(X_test)
y_pred = pd.DataFrame(y_pred)


# In[169]:


display(y_pred.head())
print(len(y_pred), '\n', y_pred[0].value_counts())


# In[170]:


ids = [str(i) for i in ids]
y_preds = [i[0] for i in y_pred.values.tolist()]
df_results = pd.DataFrame({'ID':ids, "possibility":y_preds})
df_results.to_csv('results/df_results.csv', index=None)

# In[ ]:




