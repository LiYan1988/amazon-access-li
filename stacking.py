# -*- coding: utf-8 -*-
"""
Created on Sun Oct 02 19:11:38 2016

@author: hyttring
line 34, 0.895, AUCR: 0.91598
"""

from starterPaulDuan import *
import glob

#%% load all tree models
x_train, y_train, x_test, id_test = load_data()
file_names = glob.glob('fitPredict*XT_*.pkl')
x_train_pred = []
x_test_pred = []
scores = []
for i, f in enumerate(file_names):
    t1, t2, s = read_data(f)
    x_train_pred.append(t1[:,1])
    x_test_pred.append(t2[:,1])
    scores.append(s)    

idxsort = list(np.argsort(scores)[::-1])
x_train_pred = [x_train_pred[i] for i in idxsort[:-12]]
x_test_pred = [x_test_pred[i] for i in idxsort[:-12]]
#x_train_pred = np.array(x_train_pred).T
#x_test_pred = np.array(x_test_pred).T

#%% load LR models
file_names = glob.glob('fitPredictLR_*.pkl')
for i, f in enumerate(file_names):
    t1, t2, s = read_data(f)
    if s>0.897:
        x_train_pred.append(t1[:, 1])
        x_test_pred.append(t2[:,1])
        scores.append(s)
    
x_train_pred = np.array(x_train_pred).T
x_test_pred = np.array(x_test_pred).T

# linear regression not good
#lg = linear_model.LinearRegression(fit_intercept=False, normalize=False,
#        copy_X=True)
#lg.fit(x_train_pred, y_train)
#y_pred_lg = lg.predict(x_test_pred)
#save_submission(y_pred_lg, 'submissionStackingLG.csv') # 0.89535

# linear: 0.90286/0.90315/0.90521, abs: 0.88392, sqrt: 0.89687
aucr = AUCRegressor()
aucr.fit(x_train_pred, y_train)
y_pred_aucr = aucr.predict(x_test_pred)
save_submission(y_pred_aucr, 'submissionStackingAUCR.csv') 

# 0.90385/0.90430
mlr = MLR()
mlr.fit(x_train_pred, y_train)
y_pred_mlr = mlr.predict(x_test_pred)
save_submission(y_pred_mlr, 'submissionStackingMLR.csv') 


    
