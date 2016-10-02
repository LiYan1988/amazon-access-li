# -*- coding: utf-8 -*-
"""
Created on Sun Oct 02 20:17:18 2016

@author: hyttring

fit and predict 
"""

from starterPaulDuan import *
from ensembleFunctions import *
import glob

def cv_fp(model_fit, x_train, x_test, y_train, cv=3, random_state=0):
    """Fit model with cross validation
    """
    y_train_pred = np.zeros((x_train.shape[0], 1))
    y_test_pred = np.zeros((x_test.shape[0], 1))
    y_train_pred = cv_pp(model_fit, x_train, y_train, cv=cv, 
        random_state=random_state)
    cv_score = metrics.roc_auc_score(y_train, y_train_pred[:,1])
    model_fit.fit(x_train, y_train)
    y_test_pred = model_fit.predict_proba(x_test)
       
    return y_train_pred, y_test_pred, cv_score

def cv_pp(clf, X, y, cv=3, random_state=0):
    kf = cross_validation.KFold(X.shape[0], n_folds=cv, shuffle=True, 
        random_state=random_state)
    ypred = np.zeros((X.shape[0], len(np.unique(y))))
    for train_index, test_index in kf:
        clf.fit(X[train_index], y[train_index])
        ypred[test_index] = clf.predict_proba(X[test_index])
        
    return ypred

if __name__=='__main__':    
    x_train, y_train, x_test, id_test = load_data()
    cols_drop = ['ROLE_CODE']
    model_logit = linear_model.LogisticRegression(C=2.0, random_state=0, 
        max_iter=150, n_jobs=-1, verbose=10)
    x_traing, x_testg, X, _ = \
        group_data(x_train, x_test, y_train, cols_drop=['ROLE_CODE'], 
        max_degree=[2, 3, 4], cut_off=0.5, clf=None, n_features=5)
        
    file_names = glob.glob('forward_selected_features20_C*.pkl')
    features = []
    scores = []
    C = []
    for f in file_names:
        s, t = read_data(f)
        features.append(t)
        scores.append(s)
        C.append(float(f.split('_')[3].split('C')[1]))
    idxsort = np.argsort(scores)[::-1]
    features = [features[i] for i in idxsort]
    scores = [scores[i] for i in idxsort]
    C = [C[i] for i in idxsort]
    
    for j, feature in enumerate(features):
        model_logit.C = C[j]
        model_logit.verbose = 0
        model_logit.tol = 1e-4
        encorder = preprocessing.OneHotEncoder()
        x_ = []
        for f in feature:
            xtmp = np.absolute(X[f]).reshape((X.shape[0], 1))
            x_new = encorder.fit_transform(xtmp)
            x_.append(x_new)
        x_f = sparse.hstack([i for i in x_]).tocsr()
        x_traing = x_f[:x_train.shape[0]]
        x_testg = x_f[x_train.shape[0]:]
        y_train_pred_lr, y_test_pred_lr, cv_score = cv_fp(model_logit, 
            x_traing, x_testg, y_train, cv=5, random_state=0)
        save_data('fitPredictLR_{}.pkl'.format(j), (y_train_pred_lr, 
            y_test_pred_lr, cv_score))
        print 'Iteration {} finished, score: {}'.format(j, cv_score)