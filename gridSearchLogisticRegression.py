# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 23:47:07 2016

@author: lyaa

Local search for suboptimal hyperparameters
"""

from starterPaulDuan import *

if __name__ == '__main__':
#%% load data
    x_train, y_train, x_test, id_test = load_data()
    cols_drop = ['ROLE_CODE']
    x_train.drop(cols_drop, axis=1, inplace=True)
    x_test.drop(cols_drop, axis=1, inplace=True)
#    x_trainb, x_testb = create_feat_ben(x_train, x_test)
#    x_train = sparse.hstack((x_train, x_trainb.as_matrix())).toarray()
#    x_test = sparse.hstack((x_test, x_testb.as_matrix())).tocsr()
    
    SEED=0
    model_lr = linear_model.LogisticRegression(C=2.0, random_state=0)
#    params = {'C':[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5]}
    params = {'C':[4, 5, 6, 7, 8], 'penalty':['l2', 'l1']}
    
    gridcv = grid_search.GridSearchCV(model_lr, params, scoring='roc_auc', 
        cv=4, n_jobs=-1, verbose=10)
    gridcv.fit(x_trainb, y_train)
#    save_data('gridSearchLR.pkl', gridcv)
#
#    results = gridcv.grid_scores_
#    y_pred = gridcv.predict_proba(x_testb)[:,1]
#    save_submission(y_pred, 'submissionLR.csv')