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
from datetime import datetime
import math
import os
import shutil
import time
from tensorflow.keras.models import model_from_json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def make_errip_dir(err_ip):
        if os.path.exists(err_ip):
           return
        else:
           os.makedirs(err_ip)

def series_to_supervised(data, columns, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1), shift means data moves i rows
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (columns[j], i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s(t)' % (columns[j])) for j in range(n_vars)]
        else:
            names += [('%s(t+%d)' % (columns[j], i)) for j in range(n_vars)]
    # put it all together, axis =1 means join df by rows
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def load_data(file_path):
    # dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
    # dataset = read_csv(file_path, parse_dates=['date'], index_col='date', date_parser=dateparse)
    dataset = read_csv(file_path)
    dataset.dropna(axis=0, how='any', inplace=True)
    # dataset.index.name = 'date'
    return dataset

def normalize_and_make_series(dataset, look_back):
    values = dataset.values
    values = values.astype('float64')
    # normalize features
    features_predict=[ 'NVMe_from_ceph', 'NVMe_from_transfer', 'NVMe_total_util']
    y_values=dataset[features_predict].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    scaled_y= scaler.fit_transform(y_values)
    # frame as supervised learning
    column_num = dataset.columns.size
    column_names = dataset.columns.tolist()
    reframed = series_to_supervised(scaled, column_names, look_back, 4)
    # drop columns we don't want to predict, only remain cpu which we want to predict
    drop_column = []
    for i in range((look_back+1) * column_num-4, (look_back + 4) * column_num):
        drop_column.append(i)
    reframed.drop(reframed.columns[drop_column], axis=1, inplace=True)
    return reframed, scaler

def split_data(dataset, reframed, look_back):
    column_num = dataset.columns.size
    train_size = (int)(len(dataset) / 6 * 5)
    # split trainset and testset
    values = reframed.values
    train = values[:train_size, :]
    test = values[train_size:, :]
    # split into input and outputs, the last column is the value of time t and treat it as label, namely train_y/test_y
    train_X, train_y = train[:, :-3], train[:, -3:]
    test_X, test_y = test[:, :-3], test[:, -3:]

    train_X = train_X.reshape(train_X.shape[0], look_back, column_num)
    test_X = test_X.reshape(test_X.shape[0], look_back, column_num)

    return train_X, train_y, test_X, test_y

def build_model(look_back, train_X):
    acti_func = 'relu'
    neurons = 128
    loss = 'mse'
    #optimizer = 'sgd'
    #dropout = 0.4
    batch_size = 32
    optimizer = 'adam'
    model = Sequential()
    model.add(Bidirectional(LSTM(neurons,
                   activation=acti_func,
                   return_sequences=True), input_shape=(look_back, train_X.shape[2])))
    model.add(Bidirectional(LSTM(neurons,activation=acti_func)))
    model.add(Dense(3))
    model.add(Activation('linear'))
    model.compile(loss=loss, optimizer=optimizer)
    return model

def del_errip_dir(path):
    if os.path.exists(path):
       ls = os.listdir(path)
       for i in ls:
           c_path = os.path.join(path, i)
           if os.path.isdir(c_path):
              self.del_errip_dir(c_path)
           else:
              os.remove(c_path)



start_time = time.time()
path1 = "D:/N_Prediction/data/historical/rawdata/nomaldata/15s/adddata/"
pathold = 'D:/N_Prediction/model/increase_old_model'
files= os.listdir(path1)
for file in files:
    file_path = path1 + '/' + file
    name = file_path.split('/')
    num = name[-1].split('.')
    testdata_path = 'D:/N_Prediction/data/historical/testdata/' + str(num[0]) + 'stress'
    modelsave_path = 'D:/N_Prediction/model/increase_model/' + str(num[0]) +'stressmodel.h5'
    path2 = "D:/N_Prediction/model/increase_model/"
    models = os.listdir(path2)
    src = path2 + '/' + models[0]
    dst = pathold + '/' + models[0]
    print('model name: '+ models[0])
    modelread_path = path2 + '/' + models[0]
    look_back = 5
    dataset = load_data(file_path)
    print(dataset.head())

    reframed, scaler= normalize_and_make_series(dataset, look_back)
    print(reframed.head())

    train_X, train_y, test_X, test_y = split_data(dataset, reframed, look_back)

    print(train_X[:1])
    # 二维数组个数，行数，列数
    print(train_X.shape[0],train_X.shape[1], train_X.shape[2])

    batch_size = 16

    # model = build_model(look_back, train_X)
    #Incremental training requires loading the model first
    model = load_model(modelread_path)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=10, min_lr=0)
    earlystopper = EarlyStopping(monitor='val_loss',patience=20, verbose=1)
    history = model.fit(train_X, train_y, epochs=20, batch_size=batch_size, validation_data=(test_X, test_y),
                        verbose=2, shuffle=False, callbacks=[TensorBoard(log_dir='log')])
    print(history.history['loss'])
    print(history.history['val_loss'])
    #model.save('my_model.h5')
    train_predict = model.predict(train_X, batch_size)

    #每隔30min，重新训练一次模型
    test_time = 5
    test_step = int(math.ceil(len(test_X)/test_time))
    test_predict = []
    test_predict = model.predict(test_X, batch_size)

    test_X = test_X.reshape((test_X.shape[0]*look_back, test_X.shape[2]))
    test_X = test_X[:test_y.shape[0], 1:]

    # 将test set的真实值+test_X与预测值+test_X进行反归一化
    inv_y = np.c_[test_y]
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, :]
    print('The length of inv_y is %d'%(len(inv_y)))
    # invert scaling for forecast

    inv_yhat = np.c_[test_predict]
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, :]
    print('The length of inv_yhat is %d'%(len(inv_yhat)))

    # calculate root mean squared error
    test_score = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test Score: %.2f RMSE' % test_score)

    conl=['NVMe_from_ceph', 'NVMe_from_transfer', 'NVMe_total_util']
    pred = pd.DataFrame(data=inv_yhat,columns=conl)
    pred.to_csv(testdata_path +'inv_yhat.csv',index=None)

    pred = pd.DataFrame(data=inv_y,columns=conl)
    pred.to_csv(testdata_path +'inv_y.csv',index=None)

    end_time = time.time()

    print('whole time:' + str(end_time - start_time))

    try:
        # serialize model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        # model.save_weights("model2.h5")
        #move old model
        shutil.move(src, dst)
        model.save(modelsave_path)
        print("Saved model to disk")
    except:
        parallel_model.save_weights('my_model_weights.h5')

