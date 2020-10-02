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

    index = np.array([i for i in range(0, 404)])
    newfind = np.c_[y_pred, index]

    findmax = newfind[np.argsort(-newfind[:, 0])]

    minpackloss = np.amin(findmax[:20, 1])
    toppackloss = findmax[0, 1]

    maxthroughput = findmax[0, 0]
    maxindex = findmax[0, 2]

    minpackindex = np.where(findmax[:20, 1] == np.amin(findmax[:20, 1]))[0][0]
    matchthroughput = findmax[minpackindex, 0]
    matchindex = findmax[minpackindex, 2]

    if (maxthroughput - matchthroughput) < 0.001:
        if (toppackloss - minpackloss) > 0.2:
            choose = [matchthroughput, minpackloss, matchindex]
    else:
        choose = [maxthroughput, toppackloss, maxindex]

    max_indtp = choose[2]

    max_indt = np.where(y_pred[:, 0] == np.amax(y_pred[:, 0]))[0][0]
    
    xgb_max_pred = xpred.loc[max_indtp]['num_workers']
    return xgb_max_pred

# get_N('D:/N_Prediction/data/real-time/feature/8/8_0/8_real.csv')

if __name__ == '__main__':
    path_modeldata = 'D:/N_Prediction/data/historical/rawdata/packtlossdata/all.csv'
    path_realdata = 'D:/N_Prediction/data/real-time/feature/10/10_2/10_real.csv'
    path_adddata =  'D:/N_Prediction/data/real-time/feature/10/10_2/10_reala.csv'


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

    index = np.array([i for i in range(0, 404)])
    newfind = np.c_[ypredtp,index]

    findmax = newfind[np.argsort(-newfind[:, 0])]


    minpackloss = np.amin(findmax[:20,1])
    toppackloss = findmax[0,1]

    maxthroughput = findmax[0,0]
    maxindex = findmax[0,2]

    minpackindex = np.where(findmax[:20, 1] == np.amin(findmax[:20, 1]))[0][0]
    matchthroughput = findmax[minpackindex,0]
    matchindex = findmax[minpackindex,2]

    if (maxthroughput - matchthroughput) < 0.001:
        if (toppackloss - minpackloss) > 0.2:
            choose = [matchthroughput, minpackloss,matchindex]
    else:
        choose = [maxthroughput, toppackloss,maxindex]

    max_indtp = choose[2]

    max_indt = np.where(ypredtp[:, 0] == np.amax(ypredtp[:, 0]))[0][0]

    xgb_max_pred = xpred1.loc[max_indtp]['num_workers']
    print(xgb_max_pred)
