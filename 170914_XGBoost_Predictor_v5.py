#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Added_Features: transaction_year, transaction_month, transaction_quarter, NumberofMissingFeature

"""

def get_features(df):
    df["transactiondate"] = pd.to_datetime(df["transactiondate"])
    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = df["transactiondate"].dt.month
    df['transaction_quarter'] = df['transactiondate'].dt.quarter
    #df = df.fillna(-1.0)
    return df

import numpy as np
import pandas as pd
import xgboost as xgb
import operator

train = pd.read_csv('../input/train_2016_v2.csv')
prop = pd.read_csv('../input/properties_2016.csv')
prop['missingFeature'] = prop.isnull().sum(axis=1)
sample = pd.read_csv('../input/sample_submission.csv')

for c, dtype in zip(prop.columns, prop.dtypes):
	if dtype == np.float64:
		prop[c] = prop[c].astype(np.float32)

df_train = train.merge(prop, how='left', on='parcelid')
df_train = get_features(df_train)

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
y_train = df_train['logerror'].values
y_mean = y_train.mean()
print(x_train.shape, y_train.shape)

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

split = 72000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

"""---Model Defining---"""
params = {}
params['eta'] = 0.01
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 5
params['silent'] = 1
params['base_score'] = y_mean
params['subsample'] = 0.7
params['colsample_bytree'] = 0.6
params['colsample_bylevel'] = 0.5

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)


"""---Importance Matrix Generation---"""
importance = clf.get_fscore()
importance_df = pd.DataFrame(
    sorted(importance.items(), key=operator.itemgetter(1)), 
    columns=['feature','fscore']    )

"""---Submission Generation---"""
sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')
df_test['transactiondate'] = '2016-01-01' 
df_test = get_features(df_test)
df_test['missingFeature'] = df_test.isnull().sum(axis=1)

x_test = df_test[train_columns]
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

test_dates = ['2016-10-01','2016-11-01','2016-12-01','2017-10-01','2017-11-01','2017-12-01']
test_columns = ['201610','201611','201612','201710','201711','201712']

submission = pd.read_csv('../input/sample_submission.csv')

for i in range(len(test_dates)):
    x_test['transactiondate'] = test_dates[i]
    x_test = get_features(x_test)
    x_test = x_test.drop('transactiondate', axis=1)
    d_test = xgb.DMatrix(x_test)
    p_test = clf.predict(d_test)
    submission[test_columns[i]] = [float(format(x, '.4f')) for x in p_test]
    print('predict...', i)

from datetime import datetime
submission.to_csv('../output/sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)