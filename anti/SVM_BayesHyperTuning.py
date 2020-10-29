#-*- coding: UTF-8 -*-
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
def lgb_cv(data,target,C,gamma):
    folds = KFold(n_splits=5, shuffle=True, random_state=11)
    oof = np.zeros(data.shape[0])

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(data, target)):

        X_train, X_test = np.array(data)[trn_idx], np.array(data)[val_idx]
        Y_train, Y_test = np.array(target)[trn_idx], np.array(target)[val_idx]
        from sklearn.svm import SVC

        model = SVC(C=C, gamma=gamma, probability=True)
        model.fit(X_train, Y_train)
        oof[val_idx] =model.predict(X_test)
        #del clf, trn_idx, val_idx
        del model, trn_idx, val_idx
        gc.collect()
    return -mean_squared_error(target, oof) ** 0.5
def optimize_lgb(data, target):
    def lgb_crossval(C,gamma):

        return lgb_cv(data,target,C,gamma)

    optimizer = BayesianOptimization(lgb_crossval, {

        "C":(1e-7,10),
        "gamma":(1e-7,10)
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
    C= params['C']
    gamma= params['gamma']

    param = {
        'C':C,
        'gamma':gamma

    }
    return param
