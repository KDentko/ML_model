#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('https://raw.githubusercontent.com/aczepielik/KRKtram/master/reports/report_07-23.csv')

df.delay.describe()

df.columns

df['plannedTime'] = pd.to_datetime( df['plannedTime'] )
df[['plannedTime']].info()

df['hour'] = df['plannedTime'].dt.hour


df['delay_secs'] = df['delay'].map(lambda x: x*60)
df['direction_cat'] = df['direction'].factorize()[0]
df['vehicleId'].fillna(-1, inplace = True)
df['seq_num'].fillna(-1, inplace = True)

def gen_id_num_direction(x):
    return '{} {}'.format(x['number'], x['direction'])
df['number_direction_id'] = df.apply(gen_id_num_direction, axis=1).factorize()[0]

def gen_id_stop_direction(x):
    return '{} {}'.format(x['stop'], x['direction'])
df.apply(gen_id_stop_direction, axis=1).factorize()[0]
df['stop_direction_id'] = df.apply(gen_id_stop_direction, axis=1).factorize()[0]

feats = [
    'number', 
    'stop', 
    'direction_cat', 
    'vehicleId', 
    'seq_num',
    'number_direction_id',
    'stop_direction_id',
]

X = df[ feats ].values
y = df['delay_secs'].values

#model = RandomForestRegressor(max_depth = 10, n_estimators=50, n_jobs=2)
model = DecisionTreeRegressor(max_depth = 10, random_state = 0)

scores = cross_val_score(model, X, y, cv =5, scoring = 'neg_mean_absolute_error' )
np.mean(scores), np.std(scores)





