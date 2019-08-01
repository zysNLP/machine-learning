# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def fun_vc(data, col):
    '''获取计数value_count函数
    '''
    return data[col].value_counts().reset_index().rename(columns={'index':col,col:'count_%s'%col})


def merge(df1, df2):
    '''将两个DataFrame合并为一个'''
    df_merged = pd.merge(df1, df2, left_index=True, right_index=True, how='outer')
    return df_merged


def fun_set(x):
    z = x.values.tolist()
    return len(list(set([i[0] for i in z])))

def fun_max(x):
    z = x.values.tolist()
    return max(list(set([i[0] for i in z])))

def fun_max_count(x):
    z = x.values.tolist()
    l = [i[0] for i in z]
    return l.count(max(l))

def fun_mean(x):
    z = x.values.tolist()
    l = [i[0] for i in z]
    return np.mean(l)

def fun_cont2_count(x):
    z = x.values.tolist()
    l = [i[0] for i in z]
    return len([i for i in l if i > 1])

def gpby(df, bylist, useless_col):
    '''方便好用的grouby_计数函数'''  
    gp = df.groupby(by=bylist)[[useless_col]].count().\
      reset_index().rename(index=str, columns={useless_col:'count'}).\
      sort_values(bylist, ascending=False)
    return gp

def gpby_sum(df, bylist, useful_col):
    '''方便好用的grouby_求和函数'''  
    gp = df.groupby(by=bylist)[[useful_col]].sum().\
      reset_index().rename(index=str, columns={useful_col:'%s'%useful_col+'_sum'}).\
      sort_values(bylist, ascending=False)
    return gp

def gp_byothers(df,bylist,this_other):
    return df.groupby(by=bylist)[[this_other]].apply(lambda x: fun_set(x)).\
           reset_index().rename(index=str, columns={0:(this_other+'_setNums')}).\
           sort_values(bylist, ascending=False)#[:100] #gpcol只是改名的作用

def gp_byothers_max(df,bylist,this_other):
    return df.groupby(by=bylist)[[this_other]].apply(lambda x: fun_max(x)).\
           reset_index().rename(index=str, columns={0:(this_other+'_setNums')}).\
           sort_values(bylist, ascending=False)#[:100] #gpcol只是改名的作用

 
def as_float(x):
    if x == "-":
        return 0
    else:
        return float(x)

def split_feature(feature_equal, x):
    len_camp = len(feature_equal)
    find_camp = x.find(feature_equal)
    x_camps = x[find_camp:]
    find_equal = x_camps.find('&')
    return x.replace(x,x_camps[len_camp:find_equal])


def mem_usage(pandas_obj):
    '''返回pd对象的占用内存'''
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2
    return "%.2f"%usage_mb











