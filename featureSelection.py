# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 12:12:11 2016

@author: hyttring
"""

from starterPaulDuan import *
from sklearn import feature_selection, cross_validation

x_train, y_train, x_test, id_test = load_data()
cols_drop = ['ROLE_CODE','ROLE_ROLLUP_1','ROLE_ROLLUP_2']

model_logit = linear_model.LogisticRegression(C=2.0, random_state=0, solver=
    'sag')
model_logit2 = linear_model.LogisticRegression(C=2.0, random_state=0, 
    max_iter=15, solver='sag')

x_trainh, x_testh, X, cols_good = group_data(x_train, x_test, y_train, 
            cols_drop=cols_drop, max_degree=[2, 3, 4], cut_off=2, 
            clf=None)
            
model_selection = feature_selection.SelectFromModel(model_logit, 
    threshold="mean", prefit=False)
model_selection.set_params(**{'estimator__penalty':'l1', 'estimator__n_jobs':
    -1, 'estimator__solver':'liblinear'})
model_selection.fit(x_trainh, y_train)
x_trainr = model_selection.transform(x_trainh)
cv_score = cross_validation.cross_val_score(model_logit, x_trainr, y_train, 
    cv=5, scoring='roc_auc', n_jobs=-1)
print np.mean(cv_score)