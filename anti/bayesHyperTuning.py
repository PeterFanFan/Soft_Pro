#-*- coding: UTF-8 -*-
import glob
from sklearn.svm import SVC

from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import gc
from datetime import datetime
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import fire
# -*- coding: UTF-8 -*-
import glob
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import gc
from datetime import datetime
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import fire

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    else:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
def lgb_cv(num_leaves, min_data_in_leaf, bagging_fraction, feature_fraction, lambda_l1, data, target, feature_name,
           categorical_feature):
    folds = KFold(n_splits=5, shuffle=True, random_state=11)
    oof = np.zeros(data.shape[0])
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(data, target)):
        # print(f'fold: {fold_}')
        trn_data = lgb.Dataset(data[trn_idx], label=target[trn_idx], feature_name=feature_name,
                               categorical_feature=categorical_feature)
        val_data = lgb.Dataset(data[val_idx], label=target[val_idx], feature_name=feature_name,
                               categorical_feature=categorical_feature)
        param = {
            # general parameters
            'objective': 'binary',
            'boosting': 'gbdt',
            'metric': 'rmse',
            'learning_rate': 0.01,
            # tuning parameters
            'num_leaves': int(num_leaves),
            'min_data_in_leaf': int(min_data_in_leaf),
            'bagging_freq': 1,
            'bagging_fraction': bagging_fraction,
            'feature_fraction': feature_fraction,
            'lambda_l1': lambda_l1
        }
        clf = lgb.train(param, trn_data, 10000, valid_sets=[trn_data, val_data], verbose_eval=200,
                        early_stopping_rounds=600)
#        clf = lgb.train(param, trn_data, 10000, valid_sets=[trn_data, val_data], verbose_eval=400)
        oof[val_idx] = clf.predict(data[val_idx], num_iteration=clf.best_iteration)
        del clf, trn_idx, val_idx
        gc.collect()
    return -mean_squared_error(target, oof) ** 0.5
def optimize_lgb(data, target, feature_name='auto', categorical_feature='auto'):
    def lgb_crossval(num_leaves, min_data_in_leaf, bagging_fraction, feature_fraction, lambda_l1):
        return lgb_cv(num_leaves, min_data_in_leaf, bagging_fraction, feature_fraction, lambda_l1, data, target,
                      feature_name, categorical_feature)

    optimizer = BayesianOptimization(lgb_crossval, {
        'num_leaves': (20, 200),
        'min_data_in_leaf': (10, 150),
        'bagging_fraction': (0.5, 1.0),
        'feature_fraction': (0.5, 1.0),
        'lambda_l1': (0, 10)
    })

    start_time = timer()
    optimizer.maximize(init_points=5, n_iter=100, acq='ucb', kappa=10)
    timer(start_time)
    print("Final result:", optimizer.max)
    return optimizer.max
def opt(train, target):
    result = optimize_lgb(train, target)
    print type(result)
    params = result['params']
    num_leaves = params['num_leaves']
    min_data_in_leaf = params['min_data_in_leaf']
    bagging_fraction = params['bagging_fraction']
    feature_fraction = params['feature_fraction']
    lambda_l1 = params['lambda_l1']
    param = {
        # general parameters
        'objective': 'binary',
        'boosting': 'gbdt',
        'metric': 'rmse',
        'learning_rate': 0.01,
        # tuning parameters
        'num_leaves': int(num_leaves),
        'min_data_in_leaf': int(min_data_in_leaf),
        'bagging_freq': 1,
        'bagging_fraction': bagging_fraction,
        'feature_fraction': feature_fraction,
        'lambda_l1': lambda_l1
    }
    return param
