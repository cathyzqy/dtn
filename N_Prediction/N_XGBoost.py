import shap
import math
import re
import os
import csv
from pandas import read_csv
from prometheus_http_client import Prometheus
import pandas as pd
import numpy as np
import xgboost as xgb
from http.client import HTTPException
# from lib.featureengineering import get_lag
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
import sklearn.metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import pickle

def xgb_train(x, y, xpred):
    # booster1 = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
    #                             colsample_bytree=1, max_depth=7)
    other_params = {'n_estimators': 100, 'learning_rate': 0.08, 'gamma': 0, 'subsample': 0.75,
                    'colsample_bytree': 1, 'max_depth': 7}
    booster = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', **other_params))

    train_size = (int)(len(x) / 5 * 4)

    X_train = x.iloc[:train_size, :]
    y_train = y.iloc[:train_size]
    X_test = x.iloc[train_size:, :]
    y_test= y.iloc[train_size:]

    booster.fit(X_train,y_train)

    model_socre = booster.score(X_test,y_test)
    pickle.dump(booster, open("D:/N_Prediction/model/xgboost/xgntp.dat", "wb"))


    # explainer = shap.TreeExplainer(booster)
    # shap_values = explainer.shap_values(X_test)
    # shap.summary_plot(shap_values, X_test2, plot_type="bar")


    predictiontp = booster.predict(X_test)

    score = explained_variance_score(predictiontp,y_test) #解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量
                                                         #的方差变化，值越小则说明效果越差。
    score_mae= mean_absolute_error(predictiontp,y_test)  #平均绝对误差（Mean Absolute Error，MAE），用于评估预测结果和真实数据集的接近程度的程度
                                                        #其其值越小说明拟合效果越好
    score_mse= mean_squared_error(predictiontp,y_test)    #均方差（Mean squared error，MSE），该指标计算的是拟合数据和原始数据对应样本点的误差的平方和的均值，
                                                        #其值越小说明拟合效果越好。
    scorer2 = r2_score(predictiontp,y_test)               #判定系数，其含义是也是解释回归模型的方差得分，其值取值范围是[0, 1]，越接近于1说明自变量越能解释因
                                                       #变量的方差变化，值越小则说明效果越差。

    rmse_score = mean_squared_error(y_test, predictiontp, squared=False)

    y_pred = booster.predict(xpred)
    return rmse_score, y_pred

def load_data(file_path):

    dataset = read_csv(file_path)
    dataset.dropna(axis=0, how='any', inplace=True)

    return dataset

def add_num(datapath,datasave):

    dataadd = pd.read_csv(datapath)
    features_t = ['NVMe_from_ceph','NVMe_from_transfer','NVMe_total_util']
    datanew = dataadd

    i = 4;
    while(i < 404):
        for j in range(0,4):
            datanew.loc[i] = dataadd.loc[j]
            i = i+1
    dataadd = dataadd[features_t]
    num = datapath.split('/')
    name = num[-1].split('.')
    path = 'D:/N_Prediction/data/real-time/feature/'+num[-3]+'/'+num[-2] +'/'+ str(name[0])+'r.csv'
    dataadd.to_csv(path, index=False)

    with open(path, 'r') as fin, open(datasave, 'w+', newline='') as fout:
        reader = csv.reader(fin, skipinitialspace=True)
        writer = csv.writer(fout, delimiter=',')
        count_num = 0
        for j, row in enumerate(reader):
            if j==0:
                row.append('num_workers')
                writer.writerow(row)
            if j>0:
                row.append(math.ceil(float(count_num)/4))
                writer.writerow(row)
            count_num = count_num+1

    fin.close()
    fout.close()

def get_N(real_data):
    path = real_data.split('.')
    path_adddata = path[0] + 'a.csv'
    add_num(real_data, path_adddata)
    xpred = load_data(path_adddata)

    model = pickle.load(open("D:/N_Prediction/model/xgboost/xgnt_packet.dat", "rb"))
    y_pred = model.predict(xpred)
    ypred_tp = y_pred[:,0]*1000000000 - y_pred[:,1]*10
    max_indtp = np.where(ypred_tp == np.amax(ypred_tp))[0][0]
    max_ind = np.where(y_pred[:, 0] == np.amax(y_pred[:, 0]))[0][0]


    xgb_max_pred = xpred.loc[max_indtp]['num_workers']
    return xgb_max_pred

# get_N('D:/N_Prediction/data/real-time/feature/8/8_0/8_real.csv')

if __name__ == '__main__':
    path_modeldata = 'D:/谷歌下载/datageneration/packtest/all.csv'
    path_realdata = 'D:/N_Prediction/data/real-time/feature/55_srealreal.csv'
    path_adddata =  'D:/N_Prediction/data/real-time/feature/8/8_3/8_reala.csv'


    add_num(path_realdata,path_adddata)
    data = load_data(path_modeldata)
    xpred = load_data(path_adddata)

    features = ['NVMe_from_ceph','NVMe_from_transfer','NVMe_total_util','num_workers']

    output = ['Goodput','Packet_losses']

    all = ['Goodput','Packet_losses','NVMe_from_ceph','NVMe_from_transfer','NVMe_total_util','num_workers']


    datar1 = data[all]

    dropline = datar1.loc[(datar1['Packet_losses'] > 10)].index

    datar1 = datar1.drop(index=dropline)

    x_s1 = datar1[features]
    xpred1 = xpred[features]
    y1 = datar1[output]

    xgb_score, ypredtp = xgb_train(x_s1, y1, xpred1)

    npacket = ypredtp[:, 0]*10000000 - ypredtp[:, 1]*10

    max_indt = np.where(ypredtp[:, 0] == np.amax(ypredtp[:, 0]))[0][0]

    max_indtp = np.where(npacket == np.amax(npacket))[0][0]
    xgb_max_pred = xpred1.loc[max_indtp]['num_workers']
    print(xgb_max_pred)