# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 09:41:38 2016

@author: li
"""

import pandas as pd
import numpy as np

Xtrain = pd.read_csv('train.csv')
Xtest = pd.read_csv('test.csv')
ytrain = Xtrain['ACTION']
Xtrain.drop(['ACTION', 'ROLE_CODE'], axis=1, inplace=True)
Xtest.drop(['id', 'ROLE_CODE'], axis=1, inplace=True)
Xall = pd.concat([Xtrain, Xtest], ignore_index=True)

#Xall['ROLE_ROLLUP_12'] = Xall.groupby(['ROLE_ROLLUP_1', 'ROLE_ROLLUP_2']).\
#    grouper.group_info[0]
#Xall.drop(['ROLE_ROLLUP_1', 'ROLE_ROLLUP_2'], axis=1, inplace=True)

columns = list(Xall.columns)

# frequency/count of each column
for col in columns:
    grouped = Xall.groupby(col, sort=False).size().\
        to_frame('cnt'+col).apply(np.log)
    Xall = pd.merge(Xall, grouped, left_on=col, right_index=True, how='left')

# resource percentage in the manager/role/...
for col in columns:
    if col=='RESOURCE':
        continue
    grouped = Xall.groupby([col, 'RESOURCE'], sort=False).size()
    grouped = grouped.div(grouped.sum(level=0), level=0).reset_index()
    grpcol = grouped.columns.values
    grpcol[2] = 'respct'+col
    grouped.columns = grpcol
    Xall = pd.merge(Xall, grouped, left_on=[col, 'RESOURCE'], 
                    right_on=[col, 'RESOURCE'], how='left')

# number of resources per manager/role/...
for col in columns:
    if col=='RESOURCE':
        continue
    grouped = Xall.groupby(col, sort=False)['RESOURCE']\
        .agg(lambda x:len(x.unique())).to_frame('resper'+col)
    Xall = pd.merge(Xall, grouped, left_on=col, right_index=True, how='left')
