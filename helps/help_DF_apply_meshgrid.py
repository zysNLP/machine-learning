# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:23:07 2019

@author: Dell
"""

#删除某列字符串中有相同元素的值
import pandas as pd
df = pd.DataFrame({'a':[1,2,3,4],'b':['aa11','aa12','aa13','aa14']})
df
   a     b
0  1  aa11
1  2  aa12
2  3  aa13
3  4  aa14
已知:'aa11'.strip('aa')输出'11'
现在想批量删除b列中的'aa'字符，下面是循环做法：
d_list = [d.strip('aa') for d in df['b']]
df.insert(2,'b1',d_list)
这是另一种方法：
df['c'] = df['b'].apply(lambda x:x.strip('aa'))

输出：
   a     b  b1
0  1  aa11  11
1  2  aa12  12
2  3  aa13  13
3  4  aa14  14

########### numpy中的meshgrid方法
https://blog.csdn.net/lllxxq141592654/article/details/81532855
import numpy
x=-3:1:3;y=-2:1:2; 

[X,Y]= meshgrid(x,y); 
　　X= 
　　-3 -2 -1 0 1 2 3 
　　-3 -2 -1 0 1 2 3 
　　-3 -2 -1 0 1 2 3 
　　-3 -2 -1 0 1 2 3 
　　-3 -2 -1 0 1 2 3 
　　Y = 
　　-2 -2 -2 -2 -2 -2 -2 
　　-1 -1 -1 -1 -1 -1 -1 
　　0 0 0 0 0 0 0 
　　1 1 1 1 1 1 1 
　　2 2 2 2 2 2 2











