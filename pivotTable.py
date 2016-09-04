# -*- coding: utf-8 -*-
"""
Spyder Editor

Learn pivot table, RESOURCE and MGR_ID as indexes
private score: 0.60898
"""

import xgboost
import functions
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

df = pd.read_csv('train.csv')
#df.drop(['ACTION', 'ROLE_CODE'], 1, inplace=True)
#mult = functions.findMultiplicity(df)
pt_indexes = ['MGR_ID', 'ROLE_TITLE']
pivot_table1 = pd.pivot_table(df, values=['ACTION'], 
                              index=pt_indexes, 
                              aggfunc=lambda x:np.mean(x), fill_value=0)
print(pivot_table1['ACTION'].value_counts().head(10))
#pivot_table1 = pivot_table1.reset_index()
#df_train = df.ix[:,pt_indexes]
#df_train = pd.merge(df_train, pivot_table1, on=pt_indexes)

#df_test = pd.read_csv('test.csv')
#df_test = df_test.ix[:,pt_indexes]
#print(df_test.shape)
#df_test = pd.merge(df_test, pivot_table1, left_on=pt_indexes,
#                   right_on=pt_indexes, how='left')
#df_test = df_test.fillna(np.mean(df['ACTION']))
#df_test = df_test.ix[:,['ACTION']]
#df_test.index += 1
#df_test.to_csv('submitPivotTable.csv', index_label='Id', header='Action')
#print(df_test.shape)