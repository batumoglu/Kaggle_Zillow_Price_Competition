# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 09:12:48 2017

@author: mesrur.boru
"""

param_test4 = {
 'colsample_bytree':[i/10.0 for i in range(6,11)]
 'subsample':[i/10.0 for i in range(6,11)]
}
gsearch4 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=140, max_depth=3, 
                                                 min_child_weight=5, gamma=0.4, subsample=1, colsample_bytree=1, 
                                                 objective= 'reg:linear', nthread=6, scale_pos_weight=1, seed=27), 
param_grid = param_test4, scoring='neg_mean_absolute_error',n_jobs=6,iid=False, cv=5)
gsearch4.fit(x_train,y_train)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_