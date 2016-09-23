# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:41:29 2016

@author: benbjo
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 10:39:00 2016

@author: MULTIPOS

Follow Paul Duan's starter code.

Ben's idea: 
1. frequency of each base feature value (e.g., how many times a manager/role 
    title occurs in the data set)
2. frequency of applying for the resource within the base feature value group
    (e.g., how often is the resource being aplied among all the applications 
    under the same manager/role department)
3. number of resources being applied under each manager/role department

Paul's idea:
1. cross tables of each base feature pair
2. some simple algorithmic calculation between the cross table features (e.g.,
    division, multiplication, square, cubic, log, normalization...)

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
#%% naive bayes
    x_train, y_train, x_test, id_test = load_data()
    cols_drop = ['ROLE_CODE','ROLE_ROLLUP_1','ROLE_ROLLUP_2']
    
    nb = naive_bayes.BernoulliNB(alpha=0.01)
    x_train, x_test, _ = group_data(x_train, x_test, y_train, 
        cols_drop=['ROLE_CODE'], max_degree=[2,3], cut_off=0.95)

# CV 
#    cv_score = cross_validation.cross_val_score(nb, x_train, y_train, 
#        cv=5, scoring='roc_auc', n_jobs=-1)
#    print np.mean(cv_score)
#    nb.fit(x_train, y_train)
#    y_pred = nb.predict_proba(x_test)[:,1]

# Grid search
    param = {'alpha':np.arange(0.000,0.3,0.01)}
    gridcv = grid_search.GridSearchCV(nb, param, scoring='roc_auc', cv=5, 
        verbose=3, refit=True, n_jobs=-1)
    gridcv.fit(x_train, y_train)
    print gridcv.best_score_
    y_pred = gridcv.predict_proba(x_test)[:,1]

# Save prediction
    save_submission(y_pred, 'submissionNB.csv')
