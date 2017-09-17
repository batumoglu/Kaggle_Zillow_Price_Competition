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

#properties['missingFeature'] = properties.isnull().sum(axis=1)

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

#error in calculation of the finished living area of home
train_df1['N-LivingAreaError'] = train_df1['calculatedfinishedsquarefeet']/train_df1['finishedsquarefeet12']

#proportion of living area
train_df1['N-LivingAreaProp'] = train_df1['calculatedfinishedsquarefeet']/train_df1['lotsizesquarefeet']
train_df1['N-LivingAreaProp2'] = train_df1['finishedsquarefeet12']/train_df1['finishedsquarefeet15']

#Amout of extra spacetrain_df1
train_df1['N-ExtraSpace'] = train_df1['lotsizesquarefeet'] - train_df1['calculatedfinishedsquarefeet'] 
train_df1['N-ExtraSpace-2'] = train_df1['finishedsquarefeet15'] - train_df1['finishedsquarefeet12'] 

#Total number of rooms
train_df1['N-TotalRooms'] = train_df1['bathroomcnt']*train_df1['bedroomcnt']

#Average room size
train_df1['N-AvRoomSize'] = train_df1['calculatedfinishedsquarefeet']/train_df1['roomcnt'] 

# Number of Extra rooms
train_df1['N-ExtraRooms'] = train_df1['roomcnt'] - train_df1['N-TotalRooms'] 

#Ratio of the built structure value to land area
train_df1['N-ValueProp'] = train_df1['structuretaxvaluedollarcnt']/train_df1['landtaxvaluedollarcnt']

#Does property have a garage, pool or hot tub and AC?
train_df1['N-GarPoolAC'] = ((train_df1['garagecarcnt']>0) & (train_df1['pooltypeid10']>0) & (train_df1['airconditioningtypeid']!=5))*1 

train_df1["N-location"] = train_df1["latitude"] + train_df1["longitude"]
train_df1["N-location-2"] = train_df1["latitude"]*train_df1["longitude"]
train_df1["N-location-2round"] = train_df1["N-location-2"].round(-4)

train_df1["N-latitude-round"] = train_df1["latitude"].round(-4)
train_df1["N-longitude-round"] = train_df1["longitude"].round(-4)

#Ratio of tax of property over parcel
train_df1['N-ValueRatio'] = train_df1['taxvaluedollarcnt']/train_df1['taxamount']

#TotalTaxScore
train_df1['N-TaxScore'] = train_df1['taxvaluedollarcnt']*train_df1['taxamount']

#polnomials of tax delinquency year
train_df1["N-taxdelinquencyyear-2"] = train_df1["taxdelinquencyyear"] ** 2
train_df1["N-taxdelinquencyyear-3"] = train_df1["taxdelinquencyyear"] ** 3

#Length of time since unpaid taxes
train_df1['N-life'] = 2018 - train_df1['taxdelinquencyyear']

#Number of properties in the zip
zip_count = train_df1['regionidzip'].value_counts().to_dict()
train_df1['N-zip_count'] = train_df1['regionidzip'].map(zip_count)

#Number of properties in the city
city_count = train_df1['regionidcity'].value_counts().to_dict()
train_df1['N-city_count'] = train_df1['regionidcity'].map(city_count)

#Number of properties in the city
region_count = train_df1['regionidcounty'].value_counts().to_dict()
train_df1['N-county_count'] = train_df1['regionidcounty'].map(city_count)

#polnomials of the variable
train_df1["N-structuretaxvaluedollarcnt-2"] = train_df1["structuretaxvaluedollarcnt"] ** 2
train_df1["N-structuretaxvaluedollarcnt-3"] = train_df1["structuretaxvaluedollarcnt"] ** 3

#Average structuretaxvaluedollarcnt by city
group = train_df1.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
train_df1['N-Avg-structuretaxvaluedollarcnt'] = train_df1['regionidcity'].map(group)

#Deviation away from average
train_df1['N-Dev-structuretaxvaluedollarcnt'] = abs((train_df1['structuretaxvaluedollarcnt'] - train_df1['N-Avg-structuretaxvaluedollarcnt']))/train_df1['N-Avg-structuretaxvaluedollarcnt']




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
params_v9['metric'] = ['mae']
params_v9['feature_fraction'] = 0.3
params_v9['num_leaves'] = 255
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
"""
Early stopping, best iteration is:
[219]   valid_0's l1: 0.0650543
"""
"""
Early stopping, best iteration is:
[217]   valid_0's l1: 0.0650441
"""
"""
Early stopping, best iteration is:
[223]   valid_0's l1: 0.0650022
"""
"""
Early stopping, best iteration is:
[261]   valid_0's l1: 0.0649945
"""
"""
Early stopping, best iteration is:
[328]   valid_0's l1: 0.0649755
"""
clf_final = lgb.train(params_v9, d_all, 328)

#Feature importance

lgb.plot_importance(clf_final, importance_type = 'gain',figsize=(40,40))
plt.show()


print("Prepare for the prediction ...")

sample = pd.read_csv('sample_submission.csv')

sample['parcelid'] = sample['ParcelId']

df_test = sample.merge(properties, on='parcelid', how='left')

df_test['N-LivingAreaError'] = df_test['calculatedfinishedsquarefeet']/df_test['finishedsquarefeet12']

#proportion of living area
df_test['N-LivingAreaProp'] = df_test['calculatedfinishedsquarefeet']/df_test['lotsizesquarefeet']
df_test['N-LivingAreaProp2'] = df_test['finishedsquarefeet12']/df_test['finishedsquarefeet15']

#Amout of extra spacedf_test
df_test['N-ExtraSpace'] = df_test['lotsizesquarefeet'] - df_test['calculatedfinishedsquarefeet'] 
df_test['N-ExtraSpace-2'] = df_test['finishedsquarefeet15'] - df_test['finishedsquarefeet12'] 

#Total number of rooms
df_test['N-TotalRooms'] = df_test['bathroomcnt']*df_test['bedroomcnt']

#Average room size
df_test['N-AvRoomSize'] = df_test['calculatedfinishedsquarefeet']/df_test['roomcnt'] 

# Number of Extra rooms
df_test['N-ExtraRooms'] = df_test['roomcnt'] - df_test['N-TotalRooms'] 

#Ratio of the built structure value to land area
df_test['N-ValueProp'] = df_test['structuretaxvaluedollarcnt']/df_test['landtaxvaluedollarcnt']

#Does property have a garage, pool or hot tub and AC?
df_test['N-GarPoolAC'] = ((df_test['garagecarcnt']>0) & (df_test['pooltypeid10']>0) & (df_test['airconditioningtypeid']!=5))*1 

df_test["N-location"] = df_test["latitude"] + df_test["longitude"]
df_test["N-location-2"] = df_test["latitude"]*df_test["longitude"]
df_test["N-location-2round"] = df_test["N-location-2"].round(-4)

df_test["N-latitude-round"] = df_test["latitude"].round(-4)
df_test["N-longitude-round"] = df_test["longitude"].round(-4)

#Ratio of tax of property over parcel
df_test['N-ValueRatio'] = df_test['taxvaluedollarcnt']/df_test['taxamount']

#TotalTaxScore
df_test['N-TaxScore'] = df_test['taxvaluedollarcnt']*df_test['taxamount']

#polnomials of tax delinquency year
df_test["N-taxdelinquencyyear-2"] = df_test["taxdelinquencyyear"] ** 2
df_test["N-taxdelinquencyyear-3"] = df_test["taxdelinquencyyear"] ** 3

#Length of time since unpaid taxes
df_test['N-life'] = 2018 - df_test['taxdelinquencyyear']

#Number of properties in the zip
zip_count = df_test['regionidzip'].value_counts().to_dict()
df_test['N-zip_count'] = df_test['regionidzip'].map(zip_count)

#Number of properties in the city
city_count = df_test['regionidcity'].value_counts().to_dict()
df_test['N-city_count'] = df_test['regionidcity'].map(city_count)

#Number of properties in the city
region_count = df_test['regionidcounty'].value_counts().to_dict()
df_test['N-county_count'] = df_test['regionidcounty'].map(city_count)

#polnomials of the variable
df_test["N-structuretaxvaluedollarcnt-2"] = df_test["structuretaxvaluedollarcnt"] ** 2
df_test["N-structuretaxvaluedollarcnt-3"] = df_test["structuretaxvaluedollarcnt"] ** 3

#Average structuretaxvaluedollarcnt by city
group = df_test.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()
df_test['N-Avg-structuretaxvaluedollarcnt'] = df_test['regionidcity'].map(group)

#Deviation away from average
df_test['N-Dev-structuretaxvaluedollarcnt'] = abs((df_test['structuretaxvaluedollarcnt'] - df_test['N-Avg-structuretaxvaluedollarcnt']))/df_test['N-Avg-structuretaxvaluedollarcnt']



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
del df_2016;gc.collect()
del df_2017;gc.collect()
del sample;gc.collect()

df_test_201610['ageasofdate'] = df_test_201610['year']

df_test_201610['ageasofdate'] = df_test_201610[['ageasofdate']].sub(df_test_201610['yearbuilt'], axis=0)

df_test_201710['ageasofdate'] = df_test_201710['year']

df_test_201710['ageasofdate'] = df_test_201710[['ageasofdate']].sub(df_test_201710['yearbuilt'], axis=0)

df_test_201611['ageasofdate'] = df_test_201611['year']

df_test_201611['ageasofdate'] = df_test_201611[['ageasofdate']].sub(df_test_201611['yearbuilt'], axis=0)

df_test_201711['ageasofdate'] = df_test_201711['year']

df_test_201711['ageasofdate'] = df_test_201711[['ageasofdate']].sub(df_test_201711['yearbuilt'], axis=0)

df_test_201612['ageasofdate'] = df_test_201612['year']

df_test_201612['ageasofdate'] = df_test_201612[['ageasofdate']].sub(df_test_201612['yearbuilt'], axis=0)

df_test_201712['ageasofdate'] = df_test_201712['year']

df_test_201712['ageasofdate'] = df_test_201712[['ageasofdate']].sub(df_test_201712['yearbuilt'], axis=0)


x_test201610 = df_test_201610[train_columns]

del df_test_201610;gc.collect()

for c in x_test201610.dtypes[x_test201610.dtypes == object].index.values:
    x_test201610[c] = (x_test201610[c] == True)
x_test201610 = x_test201610.values.astype(np.float32, copy=False)

x_test201611 = df_test_201611[train_columns]

del df_test_201611;gc.collect()

for c in x_test201611.dtypes[x_test201611.dtypes == object].index.values:
    x_test201611[c] = (x_test201611[c] == True)
x_test201611 = x_test201611.values.astype(np.float32, copy=False)

x_test201612 = df_test_201612[train_columns]

del df_test_201612;gc.collect()

for c in x_test201612.dtypes[x_test201612.dtypes == object].index.values:
    x_test201612[c] = (x_test201612[c] == True)
x_test201612 = x_test201612.values.astype(np.float32, copy=False)


x_test201710 = df_test_201710[train_columns]

del df_test_201710;gc.collect()

for c in x_test201710.dtypes[x_test201710.dtypes == object].index.values:
    x_test201710[c] = (x_test201710[c] == True)
x_test201710 = x_test201710.values.astype(np.float32, copy=False)

x_test201711 = df_test_201711[train_columns]

del df_test_201711;gc.collect()

for c in x_test201711.dtypes[x_test201711.dtypes == object].index.values:
    x_test201711[c] = (x_test201711[c] == True)
x_test201711 = x_test201711.values.astype(np.float32, copy=False)

x_test201712 = df_test_201712[train_columns]

del df_test_201712;gc.collect()

for c in x_test201712.dtypes[x_test201712.dtypes == object].index.values:
    x_test201712[c] = (x_test201712[c] == True)
x_test201712 = x_test201712.values.astype(np.float32, copy=False)






print("Start prediction ...")
# num_threads > 1 will predict very slow in kernel
clf_final.reset_parameter({"num_threads":1})

print("Predict 201610 ...")
p_test_201610 = clf_final.predict(x_test201710)
print("Predict 201611 ...")
p_test_201611 = clf_final.predict(x_test201711)
print("Predict 201612 ...")
p_test_201612 = clf_final.predict(x_test201712)
print("Predict 201710 ...")
p_test_201710 = clf_final.predict(x_test201710)
print("Predict 201711 ...")
p_test_201711 = clf_final.predict(x_test201711)
print("Predict 201712 ...")
p_test_201712 = clf_final.predict(x_test201712)


print("Start write result ...")
sub = pd.read_csv('sample_submission.csv')

sub[sub.columns[1]] = p_test_201610
sub[sub.columns[2]] = p_test_201611
sub[sub.columns[3]] = p_test_201612
sub[sub.columns[4]] = p_test_201710
sub[sub.columns[5]] = p_test_201711
sub[sub.columns[6]] = p_test_201712

sub.head()

sub.describe()

sub.apply(lambda x: sum(x.isnull().values), axis = 0)


from datetime import datetime
print( "\nWriting results to disk ..." )
sub.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
print( "\nFinished ...")

