# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 23:07:22 2016

@author: lyaa

fit and predict using RF XT

"""

from ensembleFunctions import *

if __name__=='__main__':
    x_train, y_train, x_test, id_test = load_data()
    x_testb = pd.read_csv('feat_ben_test.csv')
    x_trainb = pd.read_csv('feat_ben_train.csv')
    SEED=0
    params = []
    params.append({'max_depth': 30, 'min_samples_split': 9, 'n_estimators': 
        2000})
    params.append({'max_depth': 20, 'min_samples_split': 6, 'n_estimators': 
        2000})
    params.append({'min_samples_split': 3, 'n_estimators': 2000, 'max_depth': 
        30})
    params.append({'max_depth': 35, 'min_samples_split': 3, 'n_estimators': 
        3000})
    
    model_rf = ensemble.RandomForestClassifier(n_estimators=2000, 
        max_features='sqrt', max_depth=None, min_samples_split=9, 
        random_state=SEED, verbose=10, n_jobs=7)
    
    for i, param in enumerate(params):
        model_rf.set_params(**param)
        y_train_pred_xt, y_test_pred_xt, cv_score = cv_fit_predict(model_rf,
            x_trainb, x_testb, y_train, cv=5, random_state=0)
        save_data('fitPredictRFXT_{}.pkl'.format(i), 
            (y_train_pred_xt, y_test_pred_xt, cv_score))