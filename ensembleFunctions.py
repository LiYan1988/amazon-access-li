# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 11:27:03 2016

@author: lyaa

Functions for ensemble
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

def cv_fit_predict(model_fit, x_train, x_test, y_train, cv=3, random_state=0):
    """Fit model with cross validation
    """
    y_train_pred = np.zeros((x_train.shape[0], 1))
    y_test_pred = np.zeros((x_test.shape[0], 1))
    y_train_pred = cv_predict_proba(model_fit, x_train, y_train, cv=cv, 
        random_state=random_state)
    cv_score = metrics.roc_auc_score(y_train, y_train_pred[:,1])
    model_fit.fit(x_train, y_train)
    y_test_pred = model_fit.predict_proba(x_test)
    
    return y_train_pred, y_test_pred, cv_score