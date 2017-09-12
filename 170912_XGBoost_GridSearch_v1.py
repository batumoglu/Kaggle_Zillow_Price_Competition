#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:15:10 2017
@author: ozkan
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

def modelfit(alg, dtrain, target, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain, target)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='mae', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    alg.fit(dtrain, target, eval_metric='mae')
    dtrain_predictions = alg.predict(dtrain)
             
    print "\nModel Report"
    print "MAE Score (Train): %f" % metrics.mean_absolute_error(target, dtrain_predictions)
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

train = pd.read_csv("../input/train_2016_v2.csv")
properties = pd.read_csv('../input/properties_2016.csv')
print( "\nProcessing data for XGBoost ...")

for c in properties.columns:
    properties[c]=properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

train_df = train.merge(properties, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
x_test = properties.drop(['parcelid'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

xgb1 = XGBRegressor(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'reg:linear',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

modelfit(xgb1, x_train, y_train)

""" 
GridSearch Number 1
([mean: -0.06864, std: 0.00291, params: {'max_depth': 3, 'min_child_weight': 1},
  mean: -0.06861, std: 0.00300, params: {'max_depth': 3, 'min_child_weight': 3},
  mean: -0.06865, std: 0.00296, params: {'max_depth': 3, 'min_child_weight': 5},
  mean: -0.06915, std: 0.00292, params: {'max_depth': 5, 'min_child_weight': 1},
  mean: -0.06914, std: 0.00294, params: {'max_depth': 5, 'min_child_weight': 3},
  mean: -0.06911, std: 0.00293, params: {'max_depth': 5, 'min_child_weight': 5},
  mean: -0.07009, std: 0.00294, params: {'max_depth': 7, 'min_child_weight': 1},
  mean: -0.07008, std: 0.00302, params: {'max_depth': 7, 'min_child_weight': 3},
  mean: -0.06983, std: 0.00289, params: {'max_depth': 7, 'min_child_weight': 5},
  mean: -0.07115, std: 0.00290, params: {'max_depth': 9, 'min_child_weight': 1},
  mean: -0.07111, std: 0.00285, params: {'max_depth': 9, 'min_child_weight': 3},
  mean: -0.07060, std: 0.00288, params: {'max_depth': 9, 'min_child_weight': 5}],
 {'max_depth': 3, 'min_child_weight': 3},
 -0.06860758513212203)
"""
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=140, max_depth=5, 
                                                 min_child_weight=1, gamma=0, subsample=1, colsample_bytree=0.8, 
                                                 objective= 'reg:linear', nthread=6, scale_pos_weight=1, seed=27), 
param_grid = param_test1, scoring='neg_mean_absolute_error',n_jobs=6,iid=False, cv=5)
gsearch1.fit(x_train,y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

"""
GridSearch Number 2
([mean: -0.06841, std: 0.00295, params: {'max_depth': 2, 'min_child_weight': 2},
  mean: -0.06841, std: 0.00298, params: {'max_depth': 2, 'min_child_weight': 3},
  mean: -0.06843, std: 0.00300, params: {'max_depth': 2, 'min_child_weight': 4},
  mean: -0.06862, std: 0.00292, params: {'max_depth': 3, 'min_child_weight': 2},
  mean: -0.06861, std: 0.00300, params: {'max_depth': 3, 'min_child_weight': 3},
  mean: -0.06863, std: 0.00294, params: {'max_depth': 3, 'min_child_weight': 4},
  mean: -0.06890, std: 0.00296, params: {'max_depth': 4, 'min_child_weight': 2},
  mean: -0.06891, std: 0.00300, params: {'max_depth': 4, 'min_child_weight': 3},
  mean: -0.06888, std: 0.00301, params: {'max_depth': 4, 'min_child_weight': 4}],
 {'max_depth': 3, 'min_child_weight': 3},
 -0.06860758513212203)

([mean: -0.06856, std: 0.00294, params: {'max_depth': 3, 'min_child_weight': 1},
  mean: -0.06858, std: 0.00296, params: {'max_depth': 3, 'min_child_weight': 3},
  mean: -0.06856, std: 0.00291, params: {'max_depth': 3, 'min_child_weight': 5},
  mean: -0.06908, std: 0.00294, params: {'max_depth': 5, 'min_child_weight': 1},
  mean: -0.06900, std: 0.00296, params: {'max_depth': 5, 'min_child_weight': 3},
  mean: -0.06907, std: 0.00296, params: {'max_depth': 5, 'min_child_weight': 5},
  mean: -0.06989, std: 0.00295, params: {'max_depth': 7, 'min_child_weight': 1},
  mean: -0.06979, std: 0.00291, params: {'max_depth': 7, 'min_child_weight': 3},
  mean: -0.06970, std: 0.00289, params: {'max_depth': 7, 'min_child_weight': 5},
  mean: -0.07074, std: 0.00286, params: {'max_depth': 9, 'min_child_weight': 1},
  mean: -0.07078, std: 0.00279, params: {'max_depth': 9, 'min_child_weight': 3},
  mean: -0.07057, std: 0.00291, params: {'max_depth': 9, 'min_child_weight': 5}],
 {'max_depth': 3, 'min_child_weight': 5},
 -0.06855955570936204)

([mean: -0.06838, std: 0.00295, params: {'max_depth': 2, 'min_child_weight': 4},
  mean: -0.06839, std: 0.00294, params: {'max_depth': 2, 'min_child_weight': 5},
  mean: -0.06838, std: 0.00296, params: {'max_depth': 2, 'min_child_weight': 6},
  mean: -0.06859, std: 0.00295, params: {'max_depth': 3, 'min_child_weight': 4},
  mean: -0.06856, std: 0.00291, params: {'max_depth': 3, 'min_child_weight': 5},
  mean: -0.06858, std: 0.00294, params: {'max_depth': 3, 'min_child_weight': 6},
  mean: -0.06878, std: 0.00298, params: {'max_depth': 4, 'min_child_weight': 4},
  mean: -0.06883, std: 0.00297, params: {'max_depth': 4, 'min_child_weight': 5},
  mean: -0.06878, std: 0.00298, params: {'max_depth': 4, 'min_child_weight': 6}],
 {'max_depth': 3, 'min_child_weight': 5},
 -0.06855955570936204)
"""
param_test2 = {
 'max_depth':[2,3,4],
 'min_child_weight':[4,5,6]
}
gsearch2 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=140, max_depth=3, 
                                                 min_child_weight=3, gamma=0, subsample=1, colsample_bytree=0.8, 
                                                 objective= 'reg:linear', nthread=6, scale_pos_weight=1, seed=27), 
param_grid = param_test2, scoring='neg_mean_absolute_error',n_jobs=6,iid=False, cv=5)
gsearch2.fit(x_train,y_train)
gsearch2.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

"""
GridSearch Number 3
([mean: -0.06861, std: 0.00300, params: {'gamma': 0.0},
  mean: -0.06860, std: 0.00299, params: {'gamma': 0.1},
  mean: -0.06866, std: 0.00299, params: {'gamma': 0.2},
  mean: -0.06860, std: 0.00300, params: {'gamma': 0.3},
  mean: -0.06854, std: 0.00299, params: {'gamma': 0.4}],
{'gamma': 0.4},
 -0.06853986084461212)

([mean: -0.06856, std: 0.00291, params: {'gamma': 0.0},
  mean: -0.06854, std: 0.00293, params: {'gamma': 0.1},
  mean: -0.06855, std: 0.00296, params: {'gamma': 0.2},
  mean: -0.06849, std: 0.00292, params: {'gamma': 0.3},
  mean: -0.06840, std: 0.00294, params: {'gamma': 0.4}],
 {'gamma': 0.4},
 -0.06839863508939743)
"""
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=140, max_depth=3, 
                                                 min_child_weight=5, gamma=0, subsample=1, colsample_bytree=0.8, 
                                                 objective= 'reg:linear', nthread=6, scale_pos_weight=1, seed=27), 
param_grid = param_test3, scoring='neg_mean_absolute_error',n_jobs=6,iid=False, cv=5)
gsearch3.fit(x_train,y_train)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

"""
GridSearch Number 4
([mean: -0.06863, std: 0.00297, params: {'subsample': 0.6, 'colsample_bytree': 0.6},
  mean: -0.06864, std: 0.00298, params: {'subsample': 0.7, 'colsample_bytree': 0.6},
  mean: -0.06860, std: 0.00293, params: {'subsample': 0.8, 'colsample_bytree': 0.6},
  mean: -0.06856, std: 0.00294, params: {'subsample': 0.9, 'colsample_bytree': 0.6},
  mean: -0.06864, std: 0.00301, params: {'subsample': 0.6, 'colsample_bytree': 0.7},
  mean: -0.06865, std: 0.00298, params: {'subsample': 0.7, 'colsample_bytree': 0.7},
  mean: -0.06860, std: 0.00295, params: {'subsample': 0.8, 'colsample_bytree': 0.7},
  mean: -0.06859, std: 0.00294, params: {'subsample': 0.9, 'colsample_bytree': 0.7},
  mean: -0.06872, std: 0.00303, params: {'subsample': 0.6, 'colsample_bytree': 0.8},
  mean: -0.06863, std: 0.00297, params: {'subsample': 0.7, 'colsample_bytree': 0.8},
  mean: -0.06861, std: 0.00300, params: {'subsample': 0.8, 'colsample_bytree': 0.8},
  mean: -0.06857, std: 0.00297, params: {'subsample': 0.9, 'colsample_bytree': 0.8},
  mean: -0.06874, std: 0.00299, params: {'subsample': 0.6, 'colsample_bytree': 0.9},
  mean: -0.06864, std: 0.00300, params: {'subsample': 0.7, 'colsample_bytree': 0.9},
  mean: -0.06862, std: 0.00301, params: {'subsample': 0.8, 'colsample_bytree': 0.9},
  mean: -0.06858, std: 0.00294, params: {'subsample': 0.9, 'colsample_bytree': 0.9}],
 {'colsample_bytree': 0.6, 'subsample': 0.9},
 -0.06856018006801605)

([mean: -0.06856, std: 0.00295, params: {'subsample': 0.6, 'colsample_bytree': 0.6},
  mean: -0.06854, std: 0.00299, params: {'subsample': 0.7, 'colsample_bytree': 0.6},
  mean: -0.06857, std: 0.00290, params: {'subsample': 0.8, 'colsample_bytree': 0.6},
  mean: -0.06848, std: 0.00296, params: {'subsample': 0.9, 'colsample_bytree': 0.6},
  mean: -0.06860, std: 0.00301, params: {'subsample': 0.6, 'colsample_bytree': 0.7},
  mean: -0.06858, std: 0.00299, params: {'subsample': 0.7, 'colsample_bytree': 0.7},
  mean: -0.06856, std: 0.00293, params: {'subsample': 0.8, 'colsample_bytree': 0.7},
  mean: -0.06853, std: 0.00297, params: {'subsample': 0.9, 'colsample_bytree': 0.7},
  mean: -0.06865, std: 0.00299, params: {'subsample': 0.6, 'colsample_bytree': 0.8},
  mean: -0.06857, std: 0.00297, params: {'subsample': 0.7, 'colsample_bytree': 0.8},
  mean: -0.06854, std: 0.00299, params: {'subsample': 0.8, 'colsample_bytree': 0.8},
  mean: -0.06849, std: 0.00298, params: {'subsample': 0.9, 'colsample_bytree': 0.8},
  mean: -0.06866, std: 0.00300, params: {'subsample': 0.6, 'colsample_bytree': 0.9},
  mean: -0.06860, std: 0.00299, params: {'subsample': 0.7, 'colsample_bytree': 0.9},
  mean: -0.06856, std: 0.00298, params: {'subsample': 0.8, 'colsample_bytree': 0.9},
  mean: -0.06851, std: 0.00295, params: {'subsample': 0.9, 'colsample_bytree': 0.9}],
 {'colsample_bytree': 0.6, 'subsample': 0.9},
 -0.06848432421684265)

([mean: -0.06843, std: 0.00292, params: {'colsample_bytree': 0.6},
  mean: -0.06843, std: 0.00289, params: {'colsample_bytree': 0.7},
  mean: -0.06840, std: 0.00294, params: {'colsample_bytree': 0.8},
  mean: -0.06841, std: 0.00291, params: {'colsample_bytree': 0.9},
  mean: -0.06836, std: 0.00296, params: {'colsample_bytree': 1.0}],
 {'colsample_bytree': 1.0},
 -0.06836230009794235)
"""

param_test4 = {
 'colsample_bytree':[i/10.0 for i in range(6,11)]
}
gsearch4 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=140, max_depth=3, 
                                                 min_child_weight=5, gamma=0.4, subsample=1, colsample_bytree=1, 
                                                 objective= 'reg:linear', nthread=6, scale_pos_weight=1, seed=27), 
param_grid = param_test4, scoring='neg_mean_absolute_error',n_jobs=6,iid=False, cv=5)
gsearch4.fit(x_train,y_train)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

"""
GridSearch Number 5
([mean: -0.06863, std: 0.00299, params: {'subsample': 0.75, 'colsample_bytree': 0.75},
  mean: -0.06862, std: 0.00297, params: {'subsample': 0.8, 'colsample_bytree': 0.75},
  mean: -0.06861, std: 0.00297, params: {'subsample': 0.85, 'colsample_bytree': 0.75},
  mean: -0.06862, std: 0.00297, params: {'subsample': 0.75, 'colsample_bytree': 0.8},
  mean: -0.06861, std: 0.00300, params: {'subsample': 0.8, 'colsample_bytree': 0.8},
  mean: -0.06863, std: 0.00294, params: {'subsample': 0.85, 'colsample_bytree': 0.8},
  mean: -0.06864, std: 0.00295, params: {'subsample': 0.75, 'colsample_bytree': 0.85},
  mean: -0.06863, std: 0.00294, params: {'subsample': 0.8, 'colsample_bytree': 0.85},
  mean: -0.06864, std: 0.00293, params: {'subsample': 0.85, 'colsample_bytree': 0.85}],
 {'colsample_bytree': 0.8, 'subsample': 0.8},
 -0.06860758513212203)
"""
param_test5 = {
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}
gsearch5 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=140, max_depth=3, 
                                                 min_child_weight=5, gamma=0, subsample=1, colsample_bytree=1, 
                                                 objective= 'reg:linear', nthread=6, scale_pos_weight=1, seed=27), 
param_grid = param_test5, scoring='neg_mean_absolute_error',n_jobs=6,iid=False, cv=5)
gsearch5.fit(x_train,y_train)
gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_

"""
GridSearch Number 6
([mean: -0.06861, std: 0.00300, params: {'reg_alpha': 1e-05},
  mean: -0.06862, std: 0.00297, params: {'reg_alpha': 0.01},
  mean: -0.06861, std: 0.00297, params: {'reg_alpha': 0.1},
  mean: -0.06857, std: 0.00297, params: {'reg_alpha': 1},
  mean: -0.06848, std: 0.00299, params: {'reg_alpha': 100}],
 {'reg_alpha': 100},
 -0.06848281621932983)
"""
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=140, max_depth=3, 
                                                 min_child_weight=1, gamma=0.4, subsample=1, colsample_bytree=1, 
                                                 objective= 'reg:linear', nthread=6, scale_pos_weight=1, seed=27), 
param_grid = param_test6, scoring='neg_mean_absolute_error',n_jobs=6,iid=False, cv=5)
gsearch6.fit(x_train,y_train)
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_