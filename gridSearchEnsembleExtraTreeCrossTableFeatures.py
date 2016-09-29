# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 22:46:57 2016

Grid search for extra tree with cross table features

XTXT stands for extra tree cross table features
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
    x_testb = pd.read_csv('feat_ben_test.csv')
    x_trainb = pd.read_csv('feat_ben_train.csv')
    SEED=0
    model_xt = ensemble.ExtraTreesClassifier(n_estimators=2000, 
        max_features='sqrt', max_depth=None, min_samples_split=8, 
        random_state=SEED, verbose=10)
    params = {'n_estimators':[1000, 2000, 3000, 4000],
              'max_depth':[20, 30, 40, 50, 60, None], 
              'min_samples_split':[9, 15]}
    gridcv = grid_search.GridSearchCV(model_xt, params, scoring='roc_auc', 
        cv=5, n_jobs=8, verbose=10)
    gridcv.fit(x_trainb, y_train)
    save_data('gridSearchXTXT.pkl', gridcv)