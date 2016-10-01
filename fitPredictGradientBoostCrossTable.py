# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 11:22:09 2016

@author: lyaa

fit and predict using GB XT
"""

from ensembleFunctions import *

if __name__=='__main__':
    x_train, y_train, x_test, id_test = load_data()
    x_testb = pd.read_csv('feat_ben_test.csv')
    x_trainb = pd.read_csv('feat_ben_train.csv')
    SEED=0
    model_gb = ensemble.GradientBoostingClassifier(n_estimators=700, 
        learning_rate=0.01, max_depth=10, min_samples_split=6, random_state
        =SEED, verbose=10)
# cv score: 0.886977
    y_train_pred_xt, y_test_pred_xt, cv_score = cv_fit_predict(model_gb, 
        x_trainb, x_testb, y_train, cv=5, random_state=0)
    
    save_data('fitPredictGBXT.pkl', 
        (y_train_pred_xt, y_test_pred_xt, cv_score))