#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:33:35 2017

@author: mesrur
"""


import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
import gc
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
import datetime as dt
import os


os.chdir('C:\Kaggle\Zillow\Datasets')

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


monthlist = []
yearlist = []
for i in range(len(train_df)):
    month = dt.datetime.strptime(train_df['transactiondate'].values[i],'%Y-%m-%d').month
    year = dt.datetime.strptime(train_df['transactiondate'].values[i],'%Y-%m-%d').year
    monthlist.append(month) 
    yearlist.append(year)

month_df = pd.DataFrame(np.array(monthlist), columns = ['month'])
year_df = pd.DataFrame(np.array(yearlist), columns = ['year'])


train_df1=pd.concat([train_df,month_df,year_df], axis =1)

train_df1['ageasofdate'] = train_df1['year']

train_df1['ageasofdate'] = train_df1[['ageasofdate']].sub(train_df1['yearbuilt'], axis=0)


x_train = train_df1.drop(['parcelid', 'logerror','transactiondate','year'], axis=1)
y_train = train_df1["logerror"].values.astype(np.float32)

train_columns = x_train.columns

d_all = lgb.Dataset(x_train, label=y_train)


del train_df; gc.collect()
del train_df1;gc.collect()

split = 72220
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
x_train = x_train.values.astype(np.float32, copy=False)
x_valid = x_valid.values.astype(np.float32, copy=False)



d_train = lgb.Dataset(x_train, label=y_train)
d_valid = lgb.Dataset(x_valid, label=y_valid)



params_v9 = {}
params_v9['learning_rate'] = 0.03226072254727122
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
clf = lgb.train(params_v9, d_train, 100000,watchlist,early_stopping_rounds=200)

"""
Early stopping, best iteration is:
[92]    valid_0's l1: 0.0652633
"""
"""
Early stopping, best iteration is:
[210]   valid_0's l1: 0.0651442
"""
"""
Early stopping, best iteration is:
[202]   valid_0's l1: 0.0651215
"""
"""
Early stopping, best iteration is:
[197]   valid_0's l1: 0.0650868
"""
clf_final = lgb.train(params_v9, d_all, 197)

#Feature importance

lgb.plot_importance(clf_final, importance_type = 'gain',figsize=(40,40))
plt.show()


print("Prepare for the prediction ...")

sample = pd.read_csv('sample_submission.csv')

sample['parcelid'] = sample['ParcelId']

df_test = sample.merge(properties, on='parcelid', how='left')


monthlist_10 =[]
monthlist_11 =[]
monthlist_12 =[]
for i in range(len(sample)):
    month10 = int(10)
    month11 = int(11)
    month12 = int(12)
    monthlist_10.append(month10)
    monthlist_11.append(month11)
    monthlist_12.append(month12)

yearlist_2016 =[]
yearlist_2017 =[]
for i in range(len(sample)):
    year2016 = int(2016)
    year2017 = int(2017)
    yearlist_2016.append(year2016)
    yearlist_2017.append(year2017)
    
    
    
        
df_10 = pd.DataFrame(np.array(monthlist_10),columns = ['month'])
df_11 = pd.DataFrame(np.array(monthlist_11),columns = ['month'])
df_12 = pd.DataFrame(np.array(monthlist_12),columns = ['month'])
df_2016 = pd.DataFrame(np.array(yearlist_2016),columns = ['year'])
df_2017 = pd.DataFrame(np.array(yearlist_2017),columns = ['year'])



df_test_201610 = pd.concat([df_test,df_10,df_2016], axis =1)
df_test_201710 = pd.concat([df_test,df_10,df_2017], axis =1)
df_test_201611 = pd.concat([df_test,df_11,df_2016], axis =1)
df_test_201711 = pd.concat([df_test,df_11,df_2017], axis =1)
df_test_201612 = pd.concat([df_test,df_12,df_2016], axis =1)
df_test_201712 = pd.concat([df_test,df_12,df_2017], axis =1)



del df_test;gc.collect()
del properties;gc.collect()
del df_10;gc.collect()
del df_11;gc.collect()
del df_12;gc.collect()
del sample;gc.collect()

x_test10 = df_test_10[train_columns]

del df_test_10;gc.collect()

for c in x_test10.dtypes[x_test10.dtypes == object].index.values:
    x_test10[c] = (x_test10[c] == True)
x_test10 = x_test10.values.astype(np.float32, copy=False)



x_test11 = df_test_11[train_columns]

del df_test_11;gc.collect()

for c in x_test11.dtypes[x_test11.dtypes == object].index.values:
    x_test11[c] = (x_test11[c] == True)
x_test11 = x_test11.values.astype(np.float32, copy=False)

x_test12 = df_test_12[train_columns]

del df_test_12;gc.collect()

for c in x_test12.dtypes[x_test12.dtypes == object].index.values:
    x_test12[c] = (x_test12[c] == True)
x_test12 = x_test12.values.astype(np.float32, copy=False)


print("Start prediction ...")
# num_threads > 1 will predict very slow in kernel
clf_final.reset_parameter({"num_threads":1})
p_test_10 = clf_final.predict(x_test10)
p_test_11 = clf_final.predict(x_test11)
p_test_12 = clf_final.predict(x_test12)


print("Start write result ...")
sub = pd.read_csv('sample_submission.csv')

sub[sub.columns[1]] = p_test_10
sub[sub.columns[2]] = p_test_11
sub[sub.columns[3]] = p_test_12
sub[sub.columns[4]] = p_test_10
sub[sub.columns[5]] = p_test_11
sub[sub.columns[6]] = p_test_12

sub.head()



for c in sub.columns[sub.columns != 'ParcelId']:
    if sub[c] == '201610' or sub[c] == '201710':
        sub[c] = p_test_10
    elif sub[c] == '201611' or sub[c] == '201711':
        sub[c] = p_test_11
    else:
        sub[c] = p_test_12

type(sub.columns[1])

sub.info()

from datetime import datetime
print( "\nWriting results to disk ..." )
sub.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
print( "\nFinished ...")


















