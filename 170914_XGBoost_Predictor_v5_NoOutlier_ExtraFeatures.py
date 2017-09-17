#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Added_Features: transaction_year, transaction_month, transaction_quarter, NumberofMissingFeature

"""

def generate_extra_features(df_train):
    df_train['N_life'] = 2018 - df_train['yearbuilt']
    df_train['N_LivingAreaProp'] = df_train['calculatedfinishedsquarefeet']/df_train['lotsizesquarefeet']
    df_train['N_LivingAreaProp2'] = df_train['finishedsquarefeet12']/df_train['finishedsquarefeet15']
    #Amout of extra space
    df_train['N_ExtraSpace'] = df_train['lotsizesquarefeet'] - df_train['calculatedfinishedsquarefeet'] 
    df_train['N_ExtraSpace_2'] = df_train['finishedsquarefeet15'] - df_train['finishedsquarefeet12'] 
    #Total number of rooms - Keep this
    df_train['N_TotalRooms'] = df_train['bathroomcnt']*df_train['bedroomcnt']  
    df_train['N_AvRoomSize'] = df_train['calculatedfinishedsquarefeet']/df_train['N_TotalRooms']  
    df_train['N_ExtraRooms'] = df_train['roomcnt'] - df_train['N_TotalRooms']     
    #Ratio of the built structure value to land area
    df_train['N_ValueProp'] = df_train['structuretaxvaluedollarcnt']/df_train['landtaxvaluedollarcnt']    
    #Does property have a garage, pool or hot tub and AC?
    df_train['N_GarPoolAC'] = ((df_train['garagecarcnt']>0) & (df_train['pooltypeid10']>0) & (df_train['airconditioningtypeid']!=5))*1 
       
    #Ratio of tax of property over parcel
    df_train['N_ValueRatio'] = df_train['taxvaluedollarcnt']/df_train['taxamount']
    
    #TotalTaxScore
    df_train['N_TaxScore'] = df_train['taxvaluedollarcnt']*df_train['taxamount']
    
    #polnomials of tax delinquency year
    df_train["N_taxdelinquencyyear_2"] = df_train["taxdelinquencyyear"] ** 2
    df_train["N_taxdelinquencyyear_3"] = df_train["taxdelinquencyyear"] ** 3
    
    #Number of properties in the zip
    zip_count = df_train['regionidzip'].value_counts().to_dict()
    df_train['N_zip_count'] = df_train['regionidzip'].map(zip_count)
    
    #Number of properties in the city
    city_count = df_train['regionidcity'].value_counts().to_dict()
    df_train['N_city_count'] = df_train['regionidcity'].map(city_count)
    
    #Number of properties in the city
    region_count = df_train['regionidcounty'].value_counts().to_dict()
    df_train['N_county_count'] = df_train['regionidcounty'].map(region_count)
    
    #Indicator whether it has AC or not
    df_train['N_ACInd'] = (df_train['airconditioningtypeid']!=5)*1
    
    #Indicator whether it has Heating or not 
    df_train['N_HeatInd'] = (df_train['heatingorsystemtypeid']!=13)*1
    
    #There's 25 different property uses - let's compress them down to 4 categories
    df_train['N_PropType'] = df_train.propertylandusetypeid.replace({31 : "Mixed", 46 : "Other", 
            47 : "Mixed", 246 : "Mixed", 247 : "Mixed", 248 : "Mixed", 260 : "Home", 261 : "Home", 
            262 : "Home", 263 : "Home", 264 : "Home", 265 : "Home", 266 : "Home", 267 : "Home", 
            268 : "Home", 269 : "Not Built", 270 : "Home", 271 : "Home", 273 : "Home", 274 : "Other", 
            275 : "Home", 276 : "Home", 279 : "Home", 290 : "Not Built", 291 : "Not Built" })
    
    #polnomials of the variable
    df_train["N_structuretaxvaluedollarcnt_2"] = df_train["structuretaxvaluedollarcnt"] ** 2
    df_train["N_structuretaxvaluedollarcnt_3"] = df_train["structuretaxvaluedollarcnt"] ** 3
    
    #Average structuretaxvaluedollarcnt by city
    group = df_train.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
    df_train['N_Avg_structuretaxvaluedollarcnt'] = df_train['regionidcity'].map(group)
    
    #Deviation away from average
    df_train['N_Dev_structuretaxvaluedollarcnt'] = abs((df_train['structuretaxvaluedollarcnt']/df_train['N_Avg_structuretaxvaluedollarcnt']-1))
       
    return df_train
    
"""
for i in range(len(df_train.columns)):
    try:
        plt.figure(figsize=(12,8))
        type(df_train.iloc[:,i])
        sns.distplot(df_train.iloc[:,i], bins=50, kde=False)
        plt.show()
    except:
        print(df_train.columns[i])
        
    plt.figure(figsize=(12,8))
    type(df_train['taxdelinquencyyear'])
    sns.distplot(df_train['taxdelinquencyyear'], bins=50, kde=False)
    plt.show()
"""

def get_features(df):
    df["transactiondate"] = pd.to_datetime(df["transactiondate"])
    df["transaction_month"] = df["transactiondate"].dt.month
    df['transaction_quarter'] = df['transactiondate'].dt.quarter
    #df = df.fillna(-1)
    return df

def calculate_missings(df):
    missingcount = df.isnull().sum(axis=0).reset_index()
    missingcount.columns = ['column_name', 'missing_count']
    missingcount['missing_ratio'] = missingcount['missing_count']/df.shape[0]
    plt.figure( figsize = (12,6) )
    plot= sns.barplot( x = df.columns, y = missingcount['missing_count'] )
    plt.setp( plot.get_xticklabels(), rotation = 45 )    
    return missingcount 

def handle_bathrooms(df):
    df = df.drop(['calculatedbathnbr'], axis=1)
    df[['threequarterbathnbr','fullbathcnt']].fillna(0)
    return df


import numpy as np
import pandas as pd
import xgboost as xgb
import operator
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('../input/train_2016_v2.csv')
prop = pd.read_csv('../input/properties_2016.csv')
prop['missingFeature'] = prop.isnull().sum(axis=1)
sample = pd.read_csv('../input/sample_submission.csv')

for c, dtype in zip(prop.columns, prop.dtypes):
	if dtype == np.float64:
		prop[c] = prop[c].astype(np.float32)

df_train = train.merge(prop, how='left', on='parcelid')

missingcount = calculate_missings(df_train)
df_train = handle_bathrooms(df_train)
#missingbathroom = calculate_missings(df_train[['bathroomcnt','calculatedbathnbr','threequarterbathnbr','fullbathcnt']])

df_train = generate_extra_features(df_train)
df_train = get_features(df_train)


upperlevel = df_train['logerror'].mean() + df_train['logerror'].std()*2
lowerlevel = df_train['logerror'].mean() - df_train['logerror'].std()*2
df_train = df_train[df_train['logerror']<upperlevel]
df_train = df_train[df_train['logerror']>lowerlevel]

x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)
y_train = df_train['logerror'].values
y_mean = y_train.mean()

train_columns = x_train.columns

print(x_train.shape, y_train.shape)


for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

"""---Split Data as Train and Validation---"""
split = 70000
x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]
y_mean2 = y_train.mean()

"""---Model Defining---"""
d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

params = {}
params['eta'] = 0.1
params['objective'] = 'reg:linear'
params['eval_metric'] = 'rmse'
params['max_depth'] = 6
params['silent'] = 1
params['base_score'] = y_mean2
params['subsample'] = 0.8
params['colsample_bytree'] = 1
params['colsample_bylevel'] = 0.8

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

"""
for i in range(1,11):
    params = {}
    params['eta'] = 0.3
    params['objective'] = 'reg:linear'
    params['eval_metric'] = 'mae'
    params['max_depth'] = 5
    params['silent'] = 1
    params['base_score'] = y_mean
    params['subsample'] = 0.8
    params['colsample_bytree'] = 1
    params['colsample_bylevel'] = i/10.0    
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=False)
    print("Candidate Value: "+str(i)+" - Best Score: "+str(clf.best_score))
"""

"""---Importance Matrix Generation---"""
importance = clf.get_fscore()
importance_df = pd.DataFrame(
    sorted(importance.items(), key=operator.itemgetter(1)), 
    columns=['feature','fscore']    )

"""---Submission Generation---"""
sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, on='parcelid', how='left')
df_test['transactiondate'] = '2016-01-01' 
df_test = handle_bathrooms(df_test)
df_test = generate_extra_features(df_test)
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