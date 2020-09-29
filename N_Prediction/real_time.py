# -*- coding: utf-8 -*-
from math import sqrt
from numpy import concatenate
import numpy as np
#from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import tensorflow
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import optimizers
import tensorflow.keras
import keras
import pickle
from datetime import datetime
import math
import os
import time
from tensorflow.keras.models import model_from_json
os.environ["CUDA_VISIBLE_DEVICES"] = "1"




def series_to_supervised(data, columns, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    dataline = df.shape[0]
    #Choose the most recent period of time
    databegin = dataline - n_in - n_out + 1
    cols, names = list(), list()
    line = list()
    agg = list()
    result = pd.DataFrame()
    # input sequence (t-n, ... t-1), shift means data moves i rows
    for i in range(n_in, 0, -1):
        names += [('%s(t-%d)' % (columns[j], i)) for j in range(n_vars)]
    for i in range(0, n_out):
        for j in range(0, n_in):
            line = line + df.iloc[ j+i+databegin , :].tolist()
        agg.append(line)
        line = []
    result = pd.DataFrame(agg)


    result.columns = names

    return result


def load_data(file_path):

    dataset = read_csv(file_path)
    dataset.dropna(axis=0, how='any', inplace=True)

    return dataset

def normalize_and_make_series(dataset, look_back):
    values = dataset.values
    values = values.astype('float64')
    # normalize features
    features_predict = ['NVMe_from_ceph', 'NVMe_from_transfer', 'NVMe_total_util']
    y_values = dataset[features_predict].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    scaled_y = scaler.fit_transform(y_values)
    # frame as supervised learning
    column_num = dataset.columns.size
    column_names = dataset.columns.tolist()
    reframed = series_to_supervised(scaled, column_names, look_back, 4)
    return reframed, scaler

def split_data(dataset, reframed, look_back):
    column_num = dataset.columns.size
    values = reframed.values
    test_X=  values[:, :]

    test_X = test_X.reshape(test_X.shape[0], look_back, column_num)

    return  test_X

def prediction(file_path):
    name = file_path.split('/')
    num = name[-1].split('.')
    testdata_path = 'D:/N_Prediction/data/real-time/feature/'+name[-3]+'/'+name[-2] +'/'+ str(num[0]) + '_real'
    modelread_path = 'D:/N_Prediction/model/increase_model/50_9stressmodel.h5'
    look_back = 5

    dataset = load_data(file_path)
    # print(dataset.head())

    data = pd.DataFrame(dataset)
    feature = ['NVMe_from_ceph', 'NVMe_from_transfer', 'NVMe_total_util', 'CPU', 'Memory_used', 'Goodput',
               'num_workers']
    data = data[feature]
    data['Goodput'] = data['Goodput'].apply(lambda x: x / 1000000000)
    data['Memory_used'] = data['Memory_used'].apply(lambda x: x / 1000000000)
    data['NVMe_from_ceph'] = data['NVMe_from_ceph'].apply(lambda x: x / 100000000)
    data['NVMe_from_transfer'] = data['NVMe_from_transfer'].apply(lambda x: x / 1000000000)
    data = data.iloc[-8:, :]
    print(data.head())
    reframed, scaler = normalize_and_make_series(data, look_back)
    # print(reframed.head())
    test_X = split_data(data, reframed, look_back)
    model = load_model(modelread_path)
    predict = model.predict(test_X)
    inv_yhat = np.c_[predict]
    inv_yhat = scaler.inverse_transform(inv_yhat)

    col = ['NVMe_from_ceph', 'NVMe_from_transfer', 'NVMe_total_util']
    pred = pd.DataFrame(data=inv_yhat, columns=col)
    pred.to_csv(testdata_path + '.csv', index=None)
    pred['NVMe_from_ceph'] = pred['NVMe_from_ceph'].apply(lambda x: x * 100000000)
    pred['NVMe_from_transfer'] = pred['NVMe_from_transfer'].apply(lambda x: x * 1000000000)
    print(pred)

# firststarttime = datetime.now()
# prediction('D:/N_Prediction/data/real-time/collect/8/8_317/8.csv')
# firstendtime = datetime.now()
# print('first predition time:',(firstendtime-firststarttime).total_seconds(),'s')


# start_time = time.time()
# file_path = 'D:/N_Prediction/data/real-time/collect/55.csv'
# name = file_path.split('/')
# num = name[-1].split('.')
# testdata_path = 'D:/N_Prediction/data/real-time/feature/' + str(num[0]) + '_sreal'
# modelread_path = 'D:/N_Prediction/model/increase_model/99stressmodel.h5'
#
# look_back = 5
#
# dataset = load_data(file_path)
# print(dataset.head())
#
# data = pd.DataFrame(dataset)
# feature = [ 'NVMe_from_ceph', 'NVMe_from_transfer', 'NVMe_total_util','CPU','Memory_used','Goodput','num_workers']
# data = data[feature]
# data['Goodput'] = data['Goodput'].apply(lambda x: x / 1000000000)
# data['Memory_used'] = data['Memory_used'].apply(lambda x: x / 1000000000)
# data['NVMe_from_ceph'] = data['NVMe_from_ceph'].apply(lambda x: x / 100000000)
# data['NVMe_from_transfer'] = data['NVMe_from_transfer'].apply(lambda x: x / 1000000000)
# data = data.iloc[0:8, :]
# reframed, scaler = normalize_and_make_series(data, look_back)
# print(reframed.head())
#
#
# test_X = split_data(data, reframed, look_back)
#
#
# model = load_model(modelread_path)
#
# predict = model.predict(test_X)
#
# inv_yhat = np.c_[predict]
# inv_yhat = scaler.inverse_transform(inv_yhat)
# print(inv_yhat)
#
# col=['NVMe_from_ceph', 'NVMe_from_transfer', 'NVMe_total_util']
# pred = pd.DataFrame(data=inv_yhat,columns=col)
# pred.to_csv(testdata_path +'real.csv',index=None)
#
#
# end_time = time.time()
#
# print('whole time:' + str(end_time - start_time))

