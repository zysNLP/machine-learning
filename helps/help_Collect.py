# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:17:14 2019
    Python 垃圾回收机制
@author: Dell
"""
'''
python采用的是引用计数机制为主，标记-清除和分代收集两种机制为辅的策略。

引用计数

Python语言默认采用的垃圾收集机制是『引用计数法 Reference Counting』，该算法最早George E. Collins在1960的时候首次提出，50年后的今天，该算法依然被很多编程语言使用。
『引用计数法』的原理是：每个对象维护一个ob_ref字段，用来记录该对象当前被引用的次数，每当新的引用指向该对象时，它的引用计数ob_ref加1，每当该对象的引用失效时计数ob_ref减1，一旦对象的引用计数为0，该对象立即被回收，对象占用的内存空间将被释放。
它的缺点是需要额外的空间维护引用计数，这个问题是其次的，不过最主要的问题是它不能解决对象的“循环引用”，因此，也有很多语言比如Java并没有采用该算法做来垃圾的收集机制。
'''

import sys
class A():
    def __init__(self):
        '''初始化对象'''
        print('object born id:%s' %str(hex(id(self))))

def f1():
    '''循环引用变量与删除变量'''
    while True:
        c1=A()
        del c1

def func(c):
    print('obejct refcount is: ',sys.getrefcount(c)) #getrefcount()方法用于返回对象的引用计数


if __name__ == '__main__':
   #生成对象
    a=A()
    func(a)

    #增加引用
    b=a
    func(a)

    #销毁引用对象b
    del b
    func(a)














