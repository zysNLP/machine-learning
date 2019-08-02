# -*- coding: utf-8 -*-
import pandas as pd

df = pd.DataFrame({'a':[3,3,5], 'b':[2,4,6]})

df['c'] = df['a'] + df['b']

def compare(a, b):
	if a > b:
		return a
	else:
		return b

df['max'] = df.apply(lambda x: compare(x.a, x.b), axis = 1)




