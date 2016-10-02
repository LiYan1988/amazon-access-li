# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 23:40:36 2016

@author: lyaa

Forward feature selection using random logistic regression and group 3 features
"""

from starterPaulDuan import *

# 2nd group, 10 features, 20 models, 0.90706
x_train, y_train, x_test, id_test = load_data()
cols_drop = ['ROLE_CODE']
model_logit = linear_model.LogisticRegression(C=2.0, random_state=0, 
    max_iter=15, n_jobs=-1)
Y = average_models(x_train, x_test, y_train, cols_drop, [3], 3, 0, 
    model_logit, 20, 10)
y_pred = np.mean(Y,1)
save_submission(y_pred, 'submissionALRG3.csv')