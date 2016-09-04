# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 10:39:00 2016

@author: MULTIPOS

Follow Paul Duan's starter code.
"""

import pandas as pd
import numpy as np
import itertools
from scipy import sparse
import copy

import xgboost as xgb
from sklearn import (metrics, cross_validation, linear_model, preprocessing)
from sklearn import feature_selection
from sklearn import datasets


def load_data():
    """Load data
    """
    x_train = pd.read_csv('train.csv')
    y_train = x_train.pop('ACTION')
    x_test = pd.read_csv('test.csv')
    id_test = x_test.pop('id')
    return x_train, y_train, x_test, id_test

def save_submission(y_pred, file_name):
    """Save predicted data in the correct form
    """
    y_pred = pd.DataFrame(y_pred)
    y_pred.index += 1
    y_pred.index.name = 'id'
    y_pred.columns = ['ACTION']
    y_pred.to_csv(file_name)
    
def xgb_data(x_train, y_train, x_test):
    """Preprocessing for xgboost
    """
    # one hot encoder
    encoder = preprocessing.OneHotEncoder()
    X = np.vstack((x_train, x_test))
    X = encoder.fit_transform(X)
    xgbmat = xgb.DMatrix(X)
    train_index = np.arange(len(x_train.index))
    test_index = np.arange(len(x_train.index), xgbmat.num_row())
    xgbmat_train = xgbmat.slice(train_index)
    xgbmat_test = xgbmat.slice(test_index)
    xgbmat_train.set_label(y_train)
    return xgbmat_train, xgbmat_test
    
    
def process_data(x_train, x_test):
    """Process data
    """
    X = pd.concat([x_train, x_test])
    # remove redundant column
    X.drop('ROLE_CODE', 1, inplace=True)
    # add resource mgr_id pair
    X['RESOURCE_MGRID'] = X['RESOURCE']+X['MGR_ID']
    X['MGRID_RFD'] = X['MGR_ID']+X['ROLE_FAMILY_DESC']
    X['MRGID_RT'] = X['MGR_ID']+X['ROLE_TITLE']       
    X['DEPT_RT'] = X['ROLE_DEPTNAME']+X['ROLE_TITLE'] 
    X['DEPT_RTD'] = X['ROLE_DEPTNAME']+X['ROLE_FAMILY_DESC']
    X['RT_RF'] = X['ROLE_TITLE']+X['ROLE_FAMILY_DESC']
#    X['RF_RFD'] = X['ROLE_FAMILY']+X['ROLE_FAMILY_DESC']    
    # one hot encoder
    encoder = preprocessing.OneHotEncoder()
    X = encoder.fit_transform(X)
    x_train = X[:x_train.shape[0],:]
    x_test = X[x_train.shape[0]:,:]
    
    return x_train, x_test
    
def find_threshold(X, col_name, frequency, percentile):
    """Find threshold that corresponds to given percentile
    """
    th = np.arange(1, min(10, 1+np.max(frequency)))
    per = np.zeros((np.max(frequency),1))
    for i,t in enumerate(th):
        tmp = frequency[frequency<=t].index.tolist()
        per[i] = (sum(X[col_name].isin(tmp).astype(float))/
            X[col_name].shape[0])
    
    threshold = np.argmin(np.absolute(per-percentile))+1
    return threshold
    
def combine_data(X, columns, degree=2, cut_off=0.5):
    for v in itertools.combinations(columns, degree):
        col_name = '+'.join(list(v))
        X[col_name] = X.groupby(list(v)).grouper.group_info[0]
        frequency = X[col_name].value_counts()
        if cut_off<1:
            threshold = find_threshold(X, col_name, frequency, cut_off)
        else:
            threshold = int(cut_off)
        low_frequent_values = frequency[frequency<=threshold].index.tolist()
        # discard low frequency items
        X.ix[X[col_name].isin(low_frequent_values),col_name] = \
            low_frequent_values[0]
    
    return X
    
def group_data(x_train, x_test, y_train, cols_drop=['ROLE_CODE'], max_degree=2,
               cut_off=0.5, **kwargs):
    """Another data processing
    """
    X = pd.concat([x_train, x_test])
    X.drop(cols_drop, 1, inplace=True)
    columns = X.columns
    if type(max_degree)==int:
        for d in range(2, max_degree+1):
            X = combine_data(X, columns, d, cut_off)
    elif type(max_degree)==list:
        for d in max_degree:
            X = combine_data(X, columns, d, cut_off)            
            
    # feature selection, if n_features==0 skip feature selection
    if len(kwargs):
        clf = kwargs['clf']
        xt = X[:x_train.shape[0]]
        cols_good = forward_feature_selection(clf, xt, y_train)
    else:
        cols_good = X.columns
    x_train, x_test = one_hot(X, x_train, x_test, cols_good)
        

    return x_train, x_test, cols_good
    
def forward_feature_selection(clf, x, y):
    """
    According to Miroslaw's idea
    x, y are pandas dataframes
    """
    cols = list(x.columns)
    cols = list(np.random.permutation(cols))
    cols_good = []
    cols_remain = cols
    x_ = []
    score_hist = []
    
    for i, col in enumerate(cols):
        encoder = preprocessing.OneHotEncoder()
        xtmp = np.absolute(x.ix[:,col]).reshape((x.shape[0],1))
        x_new = encoder.fit_transform(xtmp)
        x_.append(x_new)
        x_features = sparse.hstack([i for i in x_]).tocsr()
        scores_cv = cross_validation.cross_val_score(clf, x_features, y, cv=10, 
                                                     scoring='roc_auc')
        score_ = np.median(scores_cv)
#        print(col,'score:',score_)
        if (len(score_hist)==0 or score_>score_hist[-1]):
            score_hist.append(score_)
            cols_good.append(col)
            cols_remain.remove(col)
    
    return cols_good
    
def one_hot(X, x_train, x_test, cols_good):
    """Encode training and testing data
    """
    encoder = preprocessing.OneHotEncoder()
    Xs = encoder.fit_transform(np.absolute(X[cols_good]))
    x_train = Xs[:x_train.shape[0],:]
    x_test = Xs[x_train.shape[0]:,:]
    
    return x_train, x_test
    
def average_models(x_train, x_test, y_train, cols_drop, max_degree,
                   cut_off, random_state, clf_select, clf_train, N):
    """Create multiple models using Miroslaw's idea:
        1. group different features, this is one of the secrets to win
        2. one hot encode
        3. forward selection of grouped feature 
        cols_drop: columns to drop, remove redundant columns
        max_degree: if integer group features up to this number, if list group
            degrees in the list
        cut_off: items whose occurances are lower than or equal to cut_off are
            dumped into one "rare category", this is one of the secrets to win
        random_state: numpy random state seed
        clf_select, clf_train: model in feature selection and training, 
            logistic regression is one of the secret to win
        N: number of models to obtain average
    """
    np.random.seed(random_state)
    Y = []
    
    for i in range(N):
        clf_select0 = copy.deepcopy(clf_select)
        clf_select0.C = np.random.rand()*3+0.5
        x_train0, x_test0, cols_good = \
            group_data(x_train, x_test, y_train, cols_drop=cols_drop, 
                       max_degree=max_degree, cut_off=1, clf=clf_select0)
        scores_cv = cross_validation.cross_val_score(clf_train,
                                                      x_train0, y_train, cv=10, 
                                                      verbose=1, 
                                                      scoring='roc_auc')
        
        print('median:',np.median(scores_cv))
        print('std:',np.std(scores_cv))
        clf_train.fit(x_train0, y_train)
        y_pred = clf_train.predict_proba(x_test0)[:,1]
        Y.append(y_pred)
    Y = np.array(Y).T
    return Y

if __name__ == '__main__':
#%% load data
    x_train, y_train, x_test, id_test = load_data()
    cols_drop = ['ROLE_CODE','ROLE_ROLLUP_1','ROLE_ROLLUP_2']
#    cols_drop = ['ROLE_CODE']
    model_logit = linear_model.LogisticRegression(C=2.0, random_state=0)
    Y = average_models(x_train, x_test, y_train, cols_drop, [2,3,4], 3, 
                       0, model_logit, model_logit, 20)
    y_pred = np.mean(Y,1)
    save_submission(y_pred, 'submissionPaulDuanLogit.csv')
#    x_train, x_test, cols_good = \
#                   group_data(x_train, x_test, y_train, cols_drop=cols_drop, 
#                   max_degree=[2,3], cut_off=1, n_features=30, random_state=0,
#                   clf=model_logit)
                   
#%% logistic regression    
#    model_logit = linear_model.LogisticRegression(C=2.0, random_state=0)
#    scores_cv = cross_validation.cross_val_score(model_logit,
#                                                  x_train, y_train, cv=10, 
#                                                  verbose=1, 
#                                                  scoring='roc_auc')
#    print(np.mean(scores_cv))
#    print(np.std(scores_cv))
#    model_logit.fit(x_train, y_train)
#    y_pred = model_logit.predict_proba(x_test)[:,1]
#    save_submission(y_pred, 'submissionPaulDuanLogit.csv')
    
#%% xgboost    
#    xgbmat_train, xgbmat_test = xgb_data(x_train, y_train, x_test)
#    param = {'booster': 'gblinear',
#         'max_depth': 20, 'learning_rate': 0.001,
#         'objective': 'binary:logistic', 'silent': 0,
#         'sample_type': 'weighted',
#         'normalize_type': 'tree',
#         'rate_drop': 0.1,'skip_drop': 0.5,'max_delta_step':5}
#    param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
#    num_round = 1000
#    res = xgb.cv(param, xgbmat_train, num_round, nfold=10, stratified=True,
#                 metrics='auc', as_pandas=True, verbose_eval=False,
#                 early_stopping_rounds=10)
#    y_pred1 = bst.predict(xgbmat_test)
#    bst = xgb.train(param, xgbmat_train, res.shape[0])
#    y_pred = bst.predict(xgbmat_test)
#    save_submission(y_pred, 'submissionPaulDuanXGB.csv')