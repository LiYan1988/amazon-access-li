# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 10:39:00 2016

@author: MULTIPOS

Follow Paul Duan's starter code.

Ben's idea: 
1. frequency of each base feature value (e.g., how many times a manager/role 
    title occurs in the data set)
2. frequency of applying for the resource within the base feature value group
    (e.g., how often is the resource being aplied among all the applications 
    under the same manager/role department)
3. number of resources being applied under each manager/role department

Paul's idea:
1. cross tables of each base feature pair
2. some simple algorithmic calculation between the cross table features (e.g.,
    division, multiplication, square, cubic, log, normalization...)

"""

import pandas as pd
import numpy as np
import itertools
from scipy import sparse
import copy

#import xgboost as xgb
from sklearn import (metrics, cross_validation, linear_model, preprocessing)
from sklearn import (feature_selection, datasets, naive_bayes, grid_search)
from sklearn import ensemble


def load_data():
    """Load data
    """
    x_train = pd.read_csv('train.csv')
    y_train = x_train.pop('ACTION')
    x_test = pd.read_csv('test.csv')
    id_test = x_test.pop('id')

    X = pd.concat([x_train, x_test])
    columns = X.columns
    # relabel all features
    for i, c in enumerate(columns):
        relabeler = preprocessing.LabelEncoder()
        X[c] = relabeler.fit_transform(X[c])
        
    x_train = X[:x_train.shape[0]]
    x_test = X[x_train.shape[0]:]
    
    return x_train, y_train, x_test, id_test

def save_submission(y_pred, file_name):
    """Save predicted data in the correct form
    """
    y_pred = pd.DataFrame(y_pred)
    y_pred.index += 1
    y_pred.index.name = 'id'
    y_pred.columns = ['ACTION']
    y_pred.to_csv(file_name)
    
#def xgb_data(x_train, y_train, x_test):
#    """Preprocessing for xgboost
#    """
#    # one hot encoder
#    encoder = preprocessing.OneHotEncoder()
#    X = np.vstack((x_train, x_test))
#    X = encoder.fit_transform(X)
#    xgbmat = xgb.DMatrix(X)
#    train_index = np.arange(len(x_train.index))
#    test_index = np.arange(len(x_train.index), xgbmat.num_row())
#    xgbmat_train = xgbmat.slice(train_index)
#    xgbmat_test = xgbmat.slice(test_index)
#    xgbmat_train.set_label(y_train)
#    return xgbmat_train, xgbmat_test
    
    
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
        relabeler = preprocessing.LabelEncoder()
        X[col_name] = relabeler.fit_transform(X[col_name])
    
    return X
    
def group_data(x_train, x_test, y_train, cols_drop=['ROLE_CODE'], max_degree=2,
               cut_off=0.5, clf=None, n_features=5):
    """Another data processing
    """
    X = pd.concat([x_train, x_test])
    X.drop(cols_drop, 1, inplace=True)
    columns = X.columns
    # relabel all features
    for i, c in enumerate(columns):
        relabeler = preprocessing.LabelEncoder()
        X[c] = relabeler.fit_transform(X[c])
    
    if type(max_degree)==int:
        for d in range(2, max_degree+1):
            X = combine_data(X, columns, d, cut_off)
    elif type(max_degree)==list:
        for d in max_degree:
            X = combine_data(X, columns, d, cut_off)            
            
    # feature selection, if n_features==0 skip feature selection
    if (clf is not None):
        xt = X[:x_train.shape[0]]
        cols_good = forward_feature_selection(clf, xt, y_train, n_features)
    else:
        cols_good = X.columns
        
    x_train, x_test = one_hot(X, x_train, x_test, cols_good)        

    return x_train, x_test, cols_good
    
def forward_feature_selection(clf, x, y, n_features):
    """
    According to Miroslaw's idea
    x, y are pandas dataframes
    n_features: maximum number of features selected
    """
    cols = list(x.columns)
    cols = list(np.random.permutation(cols))
    cols_good = []
    cols_remain = cols
    x_selected_ = []
    score_hist = []
    score_max = 0
    idx_max = 0
    cols_hist = []
    
    for j in range(len(cols)):
        score_ = []
        for col in cols_remain:
            x_ = x_selected_[:]
            encoder = preprocessing.OneHotEncoder()
            xtmp = np.absolute(x[col]).reshape((x.shape[0],1))
            x_new = encoder.fit_transform(xtmp)
            x_.append(x_new)
            x_features = sparse.hstack([i for i in x_]).tocsr()
            scores_cv = cross_validation.cross_val_score(clf, x_features, 
                y, cv=5, scoring='roc_auc', n_jobs=-1)
            score_.append((np.median(scores_cv), col, x_new))
#        print(col,'score:',np.median(scores_cv))
        score_hist.append(sorted(score_)[-1])
        if (len(score_hist)>1 and score_hist[-1][0]<score_max
            and len(cols_good)>=n_features):
            break
        else:
            col = score_hist[-1][1]
            x_new = score_hist[-1][2]
            cols_good.append(col)
            cols_remain.remove(col)
            x_selected_.append(x_new)
            score_tmp = score_hist[-1][0] if score_hist[-1][0]>score_max \
                else score_max
            idx_max = len(score_hist)-1 if score_hist[-1][0]>score_max \
                else idx_max
            score_max = score_tmp
            cols_hist.append(cols_good[:])
            print 'Iteration {}, column {} is selected, score: {}'.format(
                j, col, score_hist[-1][0])
    
    return cols_hist[idx_max]
    
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
        scores_cv = cross_validation.cross_val_score(clf_train, x_train0, 
            y_train, cv=10, verbose=1, scoring='roc_auc')
        
        print('median:',np.median(scores_cv))
        print('std:',np.std(scores_cv))
        clf_train.fit(x_train0, y_train)
        y_pred = clf_train.predict_proba(x_test0)[:,1]
        Y.append(y_pred)
    Y = np.array(Y).T
    return Y
    
def create_feat_ben(Xtrain, Xtest, keep_origin=False):
    """Create new features using Ben's method
    """
    Xall = pd.concat([Xtrain, Xtest])
    columns = list(Xall.columns)
    N = Xall.shape[0]
    
    # log count and frequency of each column
    for col in columns:
        grouped = Xall.groupby(col, sort=False).size()
        Xall = pd.merge(Xall, grouped.to_frame('logcnt'+col).apply(np.log), 
            left_on=col, right_index=True, how='left')
        Xall = pd.merge(Xall, grouped.to_frame('lincnt'+col).apply(lambda x: 
            x/N), left_on=col, right_index=True, how='left')
    
    # resource percentage in the manager/role/...
    for col in columns:
        if col=='RESOURCE':
            continue
        grouped = Xall.groupby([col, 'RESOURCE'], sort=False).size()
        grouped = grouped.div(grouped.sum(level=0), level=0).reset_index()
        grpcol = grouped.columns.values
        grpcol[2] = 'respct'+col
        grouped.columns = grpcol
        Xall = pd.merge(Xall, grouped, left_on=[col, 'RESOURCE'], 
                        right_on=[col, 'RESOURCE'], how='left')
    
    # number of resources per manager/role/...
    for col in columns:
        if col=='RESOURCE':
            continue
        grouped = Xall.groupby(col, sort=False)['RESOURCE']\
            .agg(lambda x:len(x.unique())).to_frame('resper'+col)
        Xall = pd.merge(Xall, grouped, left_on=col, right_index=True, 
                        how='left')
                        
    # cross tabulate frequencies
    for cols in itertools.combinations(columns, 2):
        grouped = Xall.groupby(by=cols, sort=False).size().apply(lambda x: 
            10.*x/N)
        grouped = grouped.to_frame('crstab'+'+'.join(cols))
        Xall = pd.merge(Xall, grouped, left_on=cols, right_index=True, 
            how='left')
                        
    if not keep_origin:
        Xall.drop(columns, axis=1, inplace=True)
    Xtrain = Xall[:Xtrain.shape[0]]
    Xtest = Xall[Xtrain.shape[0]:]
    
    return Xtrain, Xtest
    
def cv_predict_proba(clf, X, y, cv=3, random_state=0):
    kf = cross_validation.KFold(X.shape[0], n_folds=cv, shuffle=True, 
        random_state=random_state)
    ypred = np.zeros((X.shape[0], len(np.unique(y))))
    for train_index, test_index in kf:
        clf.fit(X[train_index,:], y[train_index])
        ypred[test_index] = clf.predict_proba(X[test_index])
        
    return ypred
    
def model_ensemble(models, Xtrain, ytrain, Xtest, cv=3, random_state=0):
    xpred = np.zeros((Xtrain.shape[0], len(models)*len(np.unique(ytrain))))
    for i, model in enumerate(models):
        cols = list(range(i*len(np.unique(ytrain)), 
                          (i+1)*len(np.unique(ytrain))))
        xpred[:, cols] = cv_predict_proba(model, Xtrain, ytrain, 
            cv, random_state)
        model.fit(Xtrain, ytrain)
        models[i] = model
    lg_ = linear_model.LinearRegression(fit_intercept=False, normalize=False,
        copy_X=True)
    lg_.fit(xpred, ytrain)
    
    xpred0 = np.zeros((Xtest.shape[0], len(models)*len(np.unique(ytrain))))
    for i, model in enumerate(models):
        cols = list(range(i*len(np.unique(ytrain)), 
                          (i+1)*len(np.unique(ytrain))))
        xpred0[:, cols] = model.predict_proba(Xtest.toarray())
    ypred = lg_.predict(xpred0)
    
    return ypred
        

if __name__ == '__main__':
#%% average multiple logistic regressino models
#    x_train, y_train, x_test, id_test = load_data()
#    cols_drop = ['ROLE_CODE','ROLE_ROLLUP_1','ROLE_ROLLUP_2']
#    cols_drop = ['ROLE_CODE']
#    model_logit = linear_model.LogisticRegression(C=2.0, random_state=0)
#    Y = average_models(x_train, x_test, y_train, cols_drop, [2,3,4], 3, 
#                       0, model_logit, model_logit, 20)
#    y_pred = np.mean(Y,1)
#    save_submission(y_pred, 'submissionPaulDuanLogit.csv')

#%% feature combination and logistic regression: auc = 0.908
#    x_train, y_train, x_test, id_test = load_data()
#    cols_drop = ['ROLE_CODE','ROLE_ROLLUP_1','ROLE_ROLLUP_2']
#
#    model_logit = linear_model.LogisticRegression(C=2.0, random_state=0, 
#        solver='sag')
#    model_logit2 = linear_model.LogisticRegression(C=2.0, random_state=0, 
#        solver='sag', max_iter=15)
#
#    x_trainh, x_testh, cols_good = group_data(x_train, x_test, y_train, 
#        cols_drop=cols_drop, max_degree=[2, 3, 4], cut_off=2, 
#        clf=model_logit2, n_features=40)
#    cv_score = cross_validation.cross_val_score(model_logit, x_trainh, y_train,
#        cv=5, verbose=3, scoring='roc_auc', n_jobs=-1)
#    print np.mean(cv_score)
#    model_logit.fit(x_trainh, y_train)
#    y_pred = model_logit.predict_proba(x_testh)[:,1]
#    save_submission(y_pred, 'submissionPB.csv')
    
#%% naive bayes: auc = 0.5
#    x_train, y_train, x_test, id_test = load_data()
#    cols_drop = ['ROLE_CODE']
#    x_train.drop(cols_drop, axis=1, inplace=True)
#    x_test.drop(cols_drop, axis=1, inplace=True)
#    model_nb = naive_bayes.BernoulliNB(alpha=0.03)
##    model_logit = linear_model.LogisticRegression(C=2.0, random_state=0)
#    x_trainb, x_testb = create_feat_ben(x_train, x_test)
#    x_train = sparse.hstack((x_train, x_trainb.as_matrix())).tocsr()
#    x_test = sparse.hstack((x_test, x_testb.as_matrix())).tocsr()
#    cv_score = cross_validation.cross_val_score(model_nb, x_trainb, y_train,
#        cv=5, scoring='roc_auc', n_jobs=-1)
#    params = {'alpha':np.arange(0.01, 0.1, 0.01)}
#    clf = grid_search.GridSearchCV(model_nb, params, scoring='roc_auc', cv=5, 
#        verbose=3)
#    clf.fit(x_train, y_train)
    
#%% model ensemble, auc = 0.9029 (cv=2)
#    x_train, y_train, x_test, id_test = load_data()
#    cols_drop = ['ROLE_CODE']
#    x_train.drop(cols_drop, axis=1, inplace=True)
#    x_test.drop(cols_drop, axis=1, inplace=True)
#    x_trainb, x_testb = create_feat_ben(x_train, x_test)
#    x_train = sparse.hstack((x_train, x_trainb.as_matrix())).toarray()
#    x_test = sparse.hstack((x_test, x_testb.as_matrix())).tocsr()
#    
#    SEED = 0
#    models = []
#    models.append(ensemble.RandomForestClassifier(n_estimators=2000, 
#        max_features='sqrt', max_depth=None, min_samples_split=9, 
#        random_state=SEED, verbose=10, n_jobs=-1))#8803
#    models.append(ensemble.ExtraTreesClassifier(n_estimators=2000, 
#        max_features='sqrt', max_depth=None, min_samples_split=8, 
#        random_state=SEED, verbose=10, n_jobs=-1)) #8903
#    models.append(ensemble.GradientBoostingClassifier(n_estimators=500, 
#        learning_rate=0.05, max_depth=10, min_samples_split=6, 
#        random_state=SEED, verbose=10))  #8749
#    
#    y_pred = model_ensemble(models, x_train, y_train, x_test, 
#        cv=4, random_state=0)
#    save_submission(y_pred, 'submissionEnsemble.csv')
#    
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