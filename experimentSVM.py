# -*- coding: utf-8 -*-
"""
Created on Sun Oct 02 22:38:48 2016

@author: hyttring
"""

from starterPaulDuan import *
from sklearn import svm

x_train, y_train, x_test, id_test = load_data()
cols_drop = ['ROLE_CODE']
model_logit = linear_model.LogisticRegression(C=2.0, random_state=0, 
    max_iter=150, n_jobs=-1, verbose=10)
x_traing, x_testg, X, _ = \
    group_data(x_train, x_test, y_train, cols_drop=['ROLE_CODE'], 
    max_degree=[2, 3, 4], cut_off=2, clf=None, n_features=5)
    
svc = svm.SVC()
cvclf = cross_validation.cross_val_score(svc, x_traing, y_train, cv=5, 
    scoring='roc_auc', n_jobs=1, verbose=10)
cvclf.fit(x_traing, y_train)
