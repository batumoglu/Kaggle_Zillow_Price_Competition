#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 20:27:41 2017

@author: mesrur
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
import gc
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope

print('Loading data ...')

train = pd.read_csv("train_2016_v2.csv")
properties = pd.read_csv('properties_2016.csv')



print( "\nProcessing data for light GBM ...")

for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

train_df = train.merge(properties, how='left', on='parcelid')

x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)

train_columns = x_train.columns

del train_df; gc.collect()

split = 72220
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
x_train = x_train.values.astype(np.float32, copy=False)
x_valid = x_valid.values.astype(np.float32, copy=False)

d_train = lgb.Dataset(x_train, label=y_train)
d_valid = lgb.Dataset(x_valid, label=y_valid)



#Starter LGBM = LGBM_v1

params_v1 = {}
params_v1['learning_rate'] = 0.002
params_v1['boosting_type'] = 'gbdt'
params_v1['objective'] = 'regression'
params_v1['metric'] = 'mae'
params_v1['sub_feature'] = 0.5
params_v1['num_leaves'] = 60
params_v1['min_data'] = 500
params_v1['min_hessian'] = 1
params_v1['feature_fraction_seed'] = 2
params_v1['bagging_seed'] = 3



watchlist = [d_valid]
lgb_v1 = lgb.train(params_v1, d_train, 100000,watchlist,early_stopping_rounds=200)

"""
Early stopping, best iteration is:
[535]   valid_0's l1: 0.0654404
"""
#LGBM_v2: Tune objective type 

params_v2 = {}
params_v2['learning_rate'] = 0.002
params_v2['boosting_type'] = 'gbdt'
params_v2['objective'] = 'regression_l1'
params_v2['metric'] = 'mae'
params_v2['sub_feature'] = 0.5
params_v2['num_leaves'] = 60
params_v2['min_data'] = 500
params_v2['min_hessian'] = 1
params_v2['feature_fraction_seed'] = 2
params_v2['bagging_seed'] = 3

watchlist = [d_valid]
lgb_v2 = lgb.train(params_v2, d_train, 100000,watchlist,early_stopping_rounds=200)

"""
Early stopping, best iteration is:
[3519]  valid_0's l1: 0.0653168
"""
#LGBM_v3: Tune booster type 

params_v3 = {}
params_v3['learning_rate'] = 0.002
params_v3['boosting_type'] = 'dart'
params_v3['objective'] = 'regression_l1'
params_v3['metric'] = 'mae'
params_v3['sub_feature'] = 0.5
params_v3['num_leaves'] = 60
params_v3['min_data'] = 500
params_v3['min_hessian'] = 1
params_v3['feature_fraction_seed'] = 2
params_v3['bagging_seed'] = 3

watchlist = [d_valid]
lgb_v3 = lgb.train(params_v3, d_train, 100000,watchlist,early_stopping_rounds=200)

"""
Early stopping, best iteration is:
[45]    valid_0's l1: 0.0658592
"""
#At this stage dart does not improve model so I stick with gbdt

#LGBM_v4: Tune learning rate,start with larger bins 



space_v4 ={
        'learning_rate' : hp.choice("x_learning_rate", [0.001, 0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05])
       }


def objective(space):
    params_v4 = {}
    params_v4['learning_rate'] = space['learning_rate'],
    params_v4['boosting_type'] = 'gbdt'
    params_v4['objective'] = 'regression_l1'
    params_v4['metric'] = 'mae'
    params_v4['sub_feature'] = 0.5
    params_v4['num_leaves'] = 60
    params_v4['min_data'] = 500
    params_v4['min_hessian'] = 1
    params_v4['feature_fraction_seed'] = 2
    params_v4['bagging_seed'] = 3
  
    lgb_init =lgb.train(params_v4,train_set = d_train, valid_sets =d_valid,early_stopping_rounds=200,num_boost_round=100000)
    pred = lgb_init.predict(x_valid)
    mae = mean_absolute_error(y_valid,pred)
    print "SCORE:", mae

    return{'loss':mae, 'status': STATUS_OK }


trials = Trials()

best_v4 = fmin(fn=objective,
            space=space_v4,
            algo=tpe.suggest,
            max_evals=200,
            trials=trials)

print best_v4
"""
{'x_learning_rate': 7}
"""
    params_v4 = {}
    params_v4['learning_rate'] = 0.035,
    params_v4['boosting_type'] = 'gbdt'
    params_v4['objective'] = 'regression_l1'
    params_v4['metric'] = 'mae'
    params_v4['sub_feature'] = 0.5
    params_v4['num_leaves'] = 60
    params_v4['min_data'] = 500
    params_v4['min_hessian'] = 1
    params_v4['feature_fraction_seed'] = 2
    params_v4['bagging_seed'] = 3

watchlist = [d_valid]
lgb_v4 = lgb.train(params_v4, d_train, 100000,watchlist,early_stopping_rounds=200)

"""
Early stopping, best iteration is:
[307]   valid_0's l1: 0.0652928
"""

#LGBM_v5: Tune learning rate furthermore in a small interval 



space_v5 ={
        'learning_rate' : hp.uniform("x_learning_rate", 0.03,0.04)
       }


def objective(space):
    params_v5 = {}
    params_v5['learning_rate'] = space['learning_rate'],
    params_v5['boosting_type'] = 'gbdt'
    params_v5['objective'] = 'regression_l1'
    params_v5['metric'] = 'mae'
    params_v5['sub_feature'] = 0.5
    params_v5['num_leaves'] = 60
    params_v5['min_data'] = 500
    params_v5['min_hessian'] = 1
    params_v5['feature_fraction_seed'] = 2
    params_v5['bagging_seed'] = 3
  
    lgb_init =lgb.train(params_v5,train_set = d_train, valid_sets =d_valid,early_stopping_rounds=200,num_boost_round=100000)
    pred = lgb_init.predict(x_valid)
    mae = mean_absolute_error(y_valid,pred)
    print "SCORE:", mae

    return{'loss':mae, 'status': STATUS_OK }


trials = Trials()

best_v5 = fmin(fn=objective,
            space=space_v5,
            algo=tpe.suggest,
            max_evals=200,
            trials=trials)

print best_v5
"""
{'x_learning_rate': 0.03226072254727122}
"""

    params_v5 = {}
    params_v5['learning_rate'] = 0.03226072254727122,
    params_v5['boosting_type'] = 'gbdt'
    params_v5['objective'] = 'regression_l1'
    params_v5['metric'] = 'mae'
    params_v5['feature_fraction'] = 0.5
    params_v5['num_leaves'] = 60
    params_v5['min_data'] = 500
    params_v5['min_hessian'] = 1
    params_v5['feature_fraction_seed'] = 123

watchlist = [d_valid]
lgb_v5 = lgb.train(params_v5, d_train, 10000,watchlist,early_stopping_rounds=200)

"""
Early stopping, best iteration is:
[515]   valid_0's l1: 0.0652653

"""


#LGBM_v6: Tune sub_feature  



space_v6 ={
        'sub_feature' : hp.choice("x_sub_feature", [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
       }


def objective(space):
    params_v6 = {}
    params_v6['learning_rate'] = 0.03226072254727122,
    params_v6['boosting_type'] = 'gbdt'
    params_v6['objective'] = 'regression_l1'
    params_v6['metric'] = 'mae'
    params_v6['sub_feature'] = space['sub_feature']
    params_v6['num_leaves'] = 60
    params_v6['min_data'] = 500
    params_v6['min_hessian'] = 1
    params_v6['feature_fraction_seed'] = 2
    params_v6['bagging_seed'] = 3
  
    lgb_init =lgb.train(params_v6,train_set = d_train, valid_sets =d_valid,early_stopping_rounds=200,num_boost_round=100000)
    pred = lgb_init.predict(x_valid)
    mae = mean_absolute_error(y_valid,pred)
    print "SCORE:", mae

    return{'loss':mae, 'status': STATUS_OK }


trials = Trials()

best_v6 = fmin(fn=objective,
            space=space_v6,
            algo=tpe.suggest,
            max_evals=200,
            trials=trials)

print best_v6
"""
{'x_sub_feature': 4}
"""
#does not change


#LGBM_v7: Further tune sub_feature  



space_v7 ={
        'sub_feature' : hp.uniform("x_sub_feature", 0.45,0.55)
       }


def objective(space):
    params_v7 = {}
    params_v7['learning_rate'] = 0.03226072254727122,
    params_v7['boosting_type'] = 'gbdt'
    params_v7['objective'] = 'regression_l1'
    params_v7['metric'] = 'mae'
    params_v7['sub_feature'] = space['sub_feature']
    params_v7['num_leaves'] = 60
    params_v7['min_data'] = 500
    params_v7['min_hessian'] = 1
    params_v7['feature_fraction_seed'] = 2
    params_v7['bagging_seed'] = 3
  
    lgb_init =lgb.train(params_v7,train_set = d_train, valid_sets =d_valid,early_stopping_rounds=200,num_boost_round=100000)
    pred = lgb_init.predict(x_valid)
    mae = mean_absolute_error(y_valid,pred)
    print "SCORE:", mae

    return{'loss':mae, 'status': STATUS_OK }


trials = Trials()

best_v7 = fmin(fn=objective,
            space=space_v7,
            algo=tpe.suggest,
            max_evals=200,
            trials=trials)

print best_v7
"""
{'x_sub_feature': 0.4948786511409419}
"""
    params_v7 = {}
    params_v7['learning_rate'] = 0.03226072254727122,
    params_v7['boosting_type'] = 'gbdt'
    params_v7['objective'] = 'regression_l1'
    params_v7['metric'] = 'mae'
    params_v7['feature_fraction'] = 0.4948786511409419
    params_v7['num_leaves'] = 60
    params_v7['min_data'] = 500
    params_v7['min_hessian'] = 1
    params_v7['feature_fraction_seed'] = 123

watchlist = [d_valid]
lgb_v7 = lgb.train(params_v7, d_train, 10000,watchlist,early_stopping_rounds=200)

"""
Early stopping, best iteration is:
[662]   valid_0's l1: 0.0652644
"""

#LGBM_v8: Tune num_leaves  and min data



space_v8 ={
        'num_leaves' : hp.choice("x_num_leaves", [7,15,31,63,123,255,511]),
        'min_data' : hp.choice("x_min_data", [50,100,150,200,300,400,500])
       }


def objective(space):
    params_v8 = {}
    params_v8['learning_rate'] = 0.03226072254727122,
    params_v8['boosting_type'] = 'gbdt'
    params_v8['objective'] = 'regression_l1'
    params_v8['metric'] = 'mae'
    params_v8['sub_feature'] = 0.4948786511409419
    params_v8['num_leaves'] = space['num_leaves']
    params_v8['min_data'] = space['min_data']
    params_v8['min_hessian'] = 1
    params_v8['feature_fraction_seed'] = 2
    params_v8['bagging_seed'] = 3
  
    lgb_init =lgb.train(params_v8,train_set = d_train, valid_sets =d_valid,early_stopping_rounds=200,num_boost_round=100000)
    pred = lgb_init.predict(x_valid)
    mae = mean_absolute_error(y_valid,pred)
    print "SCORE:", mae

    return{'loss':mae, 'status': STATUS_OK }


trials = Trials()

best_v8 = fmin(fn=objective,
            space=space_v8,
            algo=tpe.suggest,
            max_evals=200,
            trials=trials)

print best_v8
"""
{'x_min_data': 0, 'x_num_leaves': 6}
"""


    params_v8 = {}
    params_v8['learning_rate'] = 0.03226072254727122,
    params_v8['boosting_type'] = 'gbdt'
    params_v8['objective'] = 'regression_l1'
    params_v8['metric'] = 'mae'
    params_v8['feature_fraction'] = 0.4948786511409419
    params_v8['num_leaves'] = 511
    params_v8['min_data'] = 50
    params_v8['min_hessian'] = 1
    params_v8['feature_fraction_seed'] = 123

watchlist = [d_valid]
lgb_v8 = lgb.train(params_v8, d_train, 10000,watchlist,early_stopping_rounds=200)

"""
Early stopping, best iteration is:
[63]    valid_0's l1: 0.0652375
"""

#LGBM_v9: Tune bagging fraction 



space_v9 ={
           'bagging_fraction' : hp.choice("x_bagging_fraction", [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
       }


def objective(space):
    params_v9 = {}
    params_v9['learning_rate'] = 0.03226072254727122,
    params_v9['boosting_type'] = 'gbdt'
    params_v9['objective'] = 'regression_l1'
    params_v9['metric'] = 'mae'
    params_v9['sub_feature'] = 0.4948786511409419
    params_v9['num_leaves'] = 511
    params_v9['min_data'] = 50
    params_v9['min_hessian'] = 1
    params_v9['feature_fraction_seed'] = 2
    params_v9['bagging_seed'] = 3
    params_v9['bagging_fraction'] = space['bagging_fraction']
  
    lgb_init =lgb.train(params_v9,train_set = d_train, valid_sets =d_valid,early_stopping_rounds=200,num_boost_round=100000)
    pred = lgb_init.predict(x_valid)
    mae = mean_absolute_error(y_valid,pred)
    print "SCORE:", mae

    return{'loss':mae, 'status': STATUS_OK }


trials = Trials()

best_v9 = fmin(fn=objective,
            space=space_v9,
            algo=tpe.suggest,
            max_evals=200,
            trials=trials)

print best_v9

    params_v9 = {}
    params_v9['learning_rate'] = 0.03226072254727122,
    params_v9['boosting_type'] = 'gbdt'
    params_v9['objective'] = 'regression_l1'
    params_v9['metric'] = 'mae'
    params_v9['feature_fraction'] = 0.4948786511409419
    params_v9['num_leaves'] = 511
    params_v9['min_data'] = 50
    params_v9['min_hessian'] = 1
    params_v9['feature_fraction_seed'] = 123
    params_v9['feature_fraction_seed'] = 2
    params_v9['bagging_seed'] = 3
    params_v9['bagging_fraction'] = 0.7
    
    
    
watchlist = [d_valid]
lgb_v9 = lgb.train(params_v9, d_train, 10000,watchlist,early_stopping_rounds=200)


"""
Early stopping, best iteration is:
[92]    valid_0's l1: 0.0652633
"""





