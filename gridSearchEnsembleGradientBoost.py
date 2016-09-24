# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 22:46:57 2016

Grid search for best parameters
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
    model_rf = ensemble.GradientBoostingClassifier(n_estimators=5, 
        learning_rate=0.20, max_depth=20, min_samples_split=9, 
        random_state=SEED)
    params = {'n_estimators':[50, 100, 150, 200],
              'learning_rate':[0.01, 0.05, 0.1, 0.2], 
              'max_depth':[10, 20, 30], 'min_samples_split':[6, 9, 15]}
    
    gridcv = grid_search.GridSearchCV(model_rf, params, scoring='roc_auc', 
        cv=4, n_jobs=-1, verbose=10)
    gridcv.fit(x_train, y_train)