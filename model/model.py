# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from utils import fun_vc, merge
from features import findM, get_M, get_O, get_B

def add_features(df_data):
    
    df = df_data.copy()
    vc_dv = fun_vc(df,'dv')
    vc_dv = vc_dv.sort_values(by=['dv']).reset_index(drop=True)
    
    M, Mn = findM(vc_=vc_dv)
    #提取特征1
    df = get_M(M=M, Mn=Mn, df=df)
    #提取特征2
    df = get_O(df_data=df, Mn=Mn)
    #提取特征3
    df = get_B(df=df)
    
    df = df.reset_index(drop=True)
    df = merge(df=df)
    
    return df

def get_model_data(data, feature_cols):
    full_data = data[feature_cols].copy()
    #1、剔除Nan的数据
    full_data = full_data.dropna(axis=0)
    
    #2、拆分特征变量和目标变量
    X = full_data.drop('label', axis = 1)
    y = full_data['label']
    
    #3、将特征变量中的字符串类型转成数字类型
    X = pd.get_dummies(X)
    
    #拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 3) #默认的test_size = 0.3, 
    #print("样本总数{}".format(X.shape[0]))
    #print("训练数据样本数量{}".format(X_train.shape[0]))
    #print("测试数据样本数量{}".format(X_test.shape[0]))
    # 显示切分的结果
#    def print_labels(y_train, y_test):
#        train_0 = [t for t in y_train if t==0]
#        train_1 = [t for t in y_train if t==1]
#        test_0 = [t for t in y_test if t==0]
#        test_1 = [t for t in y_test if t==1]
#        #print('训练数据中label=0的数量:',len(train_0))
#        #print('训练数据中label=1的数量:',len(train_1))
#        #print('测试数据中label=0的数量:',len(test_0))
#        #print('测试数据中label=1的数量:',len(test_1))
#    
#    print_labels(y_train, y_test)

    return X, y, X_train, y_train, X_test, y_test