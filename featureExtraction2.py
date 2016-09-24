# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 20:32:31 2016

@author: benbjo
"""

from starterPaulDuan import *

x_train, y_train, x_test, id_test = load_data()
cols_drop = ['ROLE_CODE']
x_train.drop(cols_drop, axis=1, inplace=True)
x_test.drop(cols_drop, axis=1, inplace=True)
x_trainb, x_testb = create_feat_ben(x_train, x_test, keep_origin=True)

x = pd.concat([x_train, x_test])

#%% cross tabulate
N = x.shape[0]
columns = list(x.columns)
for cols in itertools.combinations(columns, 2):
    grouped = x.groupby(by=cols, sort=False).size().apply(lambda x: 10.*x/N)
    grouped = grouped.to_frame('crstab'+'+'.join(cols))
    x = pd.merge(x, grouped, left_on=cols, right_index=True, how='left')