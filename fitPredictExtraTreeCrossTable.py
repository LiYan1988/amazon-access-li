# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 11:22:09 2016

@author: lyaa

fit and predict using XT XT
"""

from ensembleFunctions import *

if __name__=='__main__':
    x_train, y_train, x_test, id_test = load_data()
    x_testb = pd.read_csv('feat_ben_test.csv')
    x_trainb = pd.read_csv('feat_ben_train.csv')
    SEED=0
    model_xt = ensemble.ExtraTreesClassifier(n_estimators=3000, 
        max_features='sqrt', max_depth=50, min_samples_split=9, 
        random_state=SEED, verbose=10, n_jobs=7)
# cv score: 0.886977
    y_train_pred_xt, y_test_pred_xt, cv_score = cv_fit_predict(model_xt, 
        x_trainb, x_testb, y_train, cv=5, random_state=0)
    
    save_data('fitPredictXTXT.pkl', 
        (y_train_pred_xt, y_test_pred_xt, cv_score))