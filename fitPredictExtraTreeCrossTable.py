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
        random_state=SEED, verbose=1, n_jobs=8)

    results = read_data('gridSearchXTXTResults.pkl')
    high_score = sorted(results, key=lambda x: x[1])
    low_std = sorted(results, key=lambda x: np.std(x[2]))
    score_idx = int(np.floor(len(results)*0.7))
    std_idx = int(np.ceil(len(results)*0.3))
    results_good = [x for x in results if x[1]>high_score[score_idx][1]
        and np.std(x[2])<np.std(low_std[std_idx][2])]
    results = sorted(results_good, key=lambda x: x[1])
    results.extend(high_score[-2:])
    results.extend(low_std[:2])
    params = [x[0] for x in results]
    for i, param in enumerate(params):
        if i<=4:
            pass
        else:
            model_xt.set_params(**param)
            y_train_pred_xt, y_test_pred_xt, cv_score = cv_fit_predict(model_xt, 
                x_trainb, x_testb, y_train, cv=5, random_state=0)
            save_data('fitPredictXTXT_{}.pkl'.format(i), 
                (y_train_pred_xt, y_test_pred_xt, cv_score))
