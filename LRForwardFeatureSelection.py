# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 09:14:12 2016

@author: lyaa

feature combination and logistic regression
11 features selected, auc = 0.90729
selected features saved in (the same)
LR_selected_features.pkl and
forward_selected_features_40_LogisticRegression.pkl
"""

from starterPaulDuan import *

x_train, y_train, x_test, id_test = load_data()
cols_drop = ['ROLE_CODE','ROLE_ROLLUP_1','ROLE_ROLLUP_2']

model_logit = linear_model.LogisticRegression(C=2.0, random_state=0, 
    solver='sag', n_jobs=-1)
model_logit2 = linear_model.LogisticRegression(C=2.0, random_state=0, 
    solver='sag', max_iter=15, n_jobs=-1)

x_trainh, x_testh, _, cols_good = group_data(x_train, x_test, y_train, 
    cols_drop=cols_drop, max_degree=[2, 3, 4], cut_off=2, 
    clf=model_logit2, n_features=40)
cv_score = cross_validation.cross_val_score(model_logit, x_trainh, y_train,
    cv=5, verbose=3, scoring='roc_auc', n_jobs=-1)
print np.mean(cv_score)    
model_logit.fit(x_trainh, y_train)
y_pred = model_logit.predict_proba(x_testh)[:,1]
save_submission(y_pred, 'submissionLR.csv')