# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 15:38:29 2016

@author: benbjo

Use RandomizedLogisticRegression to select grouped features
"""

from starterPaulDuan import *
from sklearn import feature_selection, cross_validation

x_train, y_train, x_test, id_test = load_data()
cols_drop = ['ROLE_CODE','ROLE_ROLLUP_1','ROLE_ROLLUP_2']

model_logit = linear_model.LogisticRegression(C=3.0, random_state=0, solver=
    'sag')
model_logit2 = linear_model.LogisticRegression(C=2.0, random_state=0, 
    max_iter=15, solver='sag')

x_trainh, x_testh, X, cols_good = group_data(x_train, x_test, y_train, 
            cols_drop=cols_drop, max_degree=[2], cut_off=2, 
            clf=None)

#randomized_logit = linear_model.RandomizedLogisticRegression(C=1.0, 
#    sample_fraction=0.5, n_resampling=250, n_jobs=1, verbose=10, 
#    selection_threshold=0.5, random_state=0)
#randomized_logit.fit(x_trainh, y_train)
#x_trainr = x_trainh[:, randomized_logit.get_support()]
#x_testr = x_testh[:, randomized_logit.get_support()]
#cv_score = cross_validation.cross_val_score(model_logit, x_trainr, y_train, 
#    cv=5, scoring='roc_auc', n_jobs=1, verbose=10)
#print np.mean(cv_score)

model_logit.fit(x_trainh, y_train)
y_pred = model_logit.predict_proba(x_testh)[:,1]
save_submission(y_pred, 'submissionRLR.csv')

