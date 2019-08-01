# -*- coding: utf-8 -*-

import paras
import pandas as pd
from utils import fun_vc

use_cols = paras.get_use_cols()
df_data = pd.read_csv('df.csv', nrows = 10000)
df_data = df_data[use_cols]
df_datah = df_data.head(10)

def get_data(df):
    
    df_using = df.copy()
    
    labels = [1 for i in range(5000)] + [0 for i in range(2000)] + \
             [1 for i in range(2000)] + [0 for i in range(1000)]
    
    #labels_train = labels[0:70000]
    #labels_test = labels[70000:]
    
    df_using.insert(len(df_using.columns), 'label', labels)
    
    return df_using

df_using = get_data(df_data)

#vc_rung = fun_vc(df_using, 'rung')

def fun_dictvc(df, usecols):
    dict_vcs = {}
    for col in usecols:
        vc = fun_vc(df, col)
        dict_vcs[col] = vc
    return dict_vcs
#dict_value_counts = fun_dictvc(df=df_using, usecols=use_cols)

def get_variance(df):
    row_count = df.shape[0]-1
    col_count = df.shape[1]-1
    v = (row_count-1)*(col_count-1)
    return v

def get_chi_square_value(df1, df2):
    df1 = df1.drop(['col_total'])
    df2 = df2.drop(['col_total'])
    del df1['row_total']
    del df2['row_total']
    mtr1 = df1.astype(int).as_matrix()
    mtr2 = df2.astype(int).as_matrix()
    mtr = ((mtr1-mtr2)**2)/mtr2
    return mtr.sum()

def get_classify(df_data, col_result, col_pred):

    df = df_data.copy()
    
    df = df.groupby([col_result, col_pred]).agg({col_result:['count']})
    df = df.reset_index()
    df.columns = [col_result,col_pred,'count']
    
    df_zero = df[df[col_result] == 0]
    df_ones = df[df[col_result] == 1]
    
    list_zero = df_zero[col_pred].values.tolist()
    list_ones = df_ones[col_pred].values.tolist()
    
    len_z = len(list_zero)
    len_o = len(list_ones)
    
    if len_z > len_o:
        inter = list(set(list_zero).difference(set(list_ones)))
    elif len_z < len_o:
        inter = list(set(list_ones).difference(set(list_zero)))
    else:
        inter = []
    
    preds = df[col_pred].values.tolist()
    if len(inter) != 0:
        for pred in preds:
            if pred == inter[0]:
                finded = preds.index(pred)
        df = df.drop(index=finded, axis=0)
    else:
        df = df
    
    df = pd.pivot_table(df, values = 'count', index=col_result, columns = col_pred).reset_index()
    df['row_total'] = df.sum(axis=1)
    df.set_index(col_result, inplace=True)
    df.loc['ratio(%)'] = df.loc[0]*100/df.loc[1]
    df = df.drop(['ratio(%)'])
    df.loc['col_total']=df.sum(axis=0)
    
    df2 = df.copy()
    total = df2[['row_total']].loc[['col_total']].values[0][0]
    for col in df2:
        df2[col] = df2[[col]].loc[['col_total']].values[0][0] * df2['row_total']/total
    df2 = df2.drop(['col_total'])
    df2.loc['col_total']=df2.sum(axis=0)
    
    x = get_chi_square_value(df,df2)#顺序：(实际df,推算df)
    v = get_variance(df2) # v=（行数-1）（列数-1）
    return x, v

def get_train(df, col):
    
    df_train = df.iloc[:70000]
    df_train = df_train[[col, 'label']]
    
    return df_train

#df_train = get_train(df_using, 'feature1')
#x, v = get_classify(df_data=df_train, col_result='label', col_pred= 'feature1')

dict_chi_squares = {}
for co in use_cols:
    df_tr = get_train(df_using, co)
    try:
        x, v = get_classify(df_tr, 'label', co)
        dict_chi_squares[co] = [x, v]
        print(co,'---good!!!')
    except ValueError:
        #x, v = 'null','null'
        #dict_chi_squares[co] = [x, v]
        print(co,'---error，该特征需要进一步分析')

print('筛选的部分有意义的特征，其卡方值和自由度')
for i in dict_chi_squares.items():
    print(i)
print('待筛选的其他特征也具有一些比较明显的意义，需要进一步分析')







