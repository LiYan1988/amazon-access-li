# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:03:33 2016

@author: benbjo

Local search for suboptimal hyperparameters
"""

import pandas as pd
import numpy as np
import itertools
from scipy import sparse
import copy

#import xgboost as xgb
from sklearn import (metrics, cross_validation, linear_model, preprocessing)
from sklearn import feature_selection, datasets, naive_bayes, grid_search
from starterPaulDuan import *

if __name__ == '__main__':
#%% load data
    x_train, y_train, x_test, id_test = load_data()
    cols_drop = ['ROLE_CODE']
    x_train.drop(cols_drop, axis=1, inplace=True)
    x_test.drop(cols_drop, axis=1, inplace=True)
    x_trainb, x_testb = create_feat_ben(x_train, x_test)
    x_train = sparse.hstack((x_train, x_trainb.as_matrix())).toarray()
    x_test = sparse.hstack((x_test, x_testb.as_matrix())).tocsr()
    
    SEED=0
    model_gb = ensemble.GradientBoostingClassifier(n_estimators=5, 
        learning_rate=0.20, max_depth=20, min_samples_split=9, 
        random_state=SEED)
    params = {'n_estimators':[400, 500, 600, 700],
          'learning_rate':[0.01, 0.03, 0.05, 0.1], 
          'min_samples_split':[6, 9, 15], 'max_depth':[10, 20, 30]}
    params = {'n_estimators':[400, 500, 600, 700],
              'learning_rate':[0.01], 
              'max_depth':[10, 30, 40], 'min_samples_split':[6]}

#    {'max_depth': 10, 'min_samples_split': 6, 'n_estimators': 400, 
#    'learning_rate'}, 0.87737
# {'learning_rate': 0.01, 'max_depth': 10, 'min_samples_split': 6, 
# 'n_estimators': 700}: 0.87884
    gridcv = grid_search.GridSearchCV(model_gb, params, scoring='roc_auc', 
        cv=4, n_jobs=-1, verbose=10)
    gridcv.fit(x_trainb, y_train)
    save_data('gridSearchGBXTLocal.pkl', gridcv)

    gridcv = read_data('gridSearchGBXTLocal.pkl')
    y_pred = gridcv.predict_proba(x_testb)[:,1]
    save_submission(y_pred, 'submissionGBXTLocal.csv')