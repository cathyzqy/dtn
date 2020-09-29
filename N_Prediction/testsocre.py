import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os
import requests
from PIL import Image

#get real-time score
path1 = 'D:/N_Prediction/data/real-time/feature/80/'
pathsave = 'D:/N_Prediction/data/real-time/feature/80/rmse.csv'
raw= pd.read_csv('D:/N_Prediction/data/compare/single/pack/80_632.csv')
dirs= os.listdir(path1)
for dir in dirs:
    path2= path1 + dir + '/80_real.csv'
    real = pd.read_csv(path2)
    feature = ['NVMe_from_ceph', 'NVMe_from_transfer', 'NVMe_total_util']
    rwadata = raw[feature].iloc[8:12, :]
    rwadata['NVMe_from_ceph'] = rwadata['NVMe_from_ceph'].apply(lambda x: x / 100000000)
    rwadata['NVMe_from_transfer'] = rwadata['NVMe_from_transfer'].apply(lambda x: x / 1000000000)
    test_score = np.sqrt(mean_squared_error(rwadata, real))
    if not os.path.isfile(pathsave):
        datarsme = pd.DataFrame(columns=['num_workers', 'rmse'],data=[[80, test_score]])
    else:
        datarsme = pd.read_csv(pathsave)
        datarsme = datarsme.append([{'num_workers': 80, 'rmse': test_score}], ignore_index=True)

    datarsme.to_csv('D:/N_Prediction/data/real-time/feature/80/rmse.csv',index=None)


#get each transfer time
# orchestrator = 'dtn-orchestrator.nautilus.optiputer.net'
# def get_transfer_init(transfer_id):
#     response = requests.get('http://{}/transfer/{}'.format(orchestrator, transfer_id))
#     result = response.json()
#     print(result)
#     data2 = [{'num_workers': int(result['num_workers']), 'time': int(result['end_time'] - result['start_time'])}]
#     data = pd.DataFrame(data2, columns=['num_workers', 'time'])
#     data.to_csv('D:/N_Prediction/data/real-time/collect/100/time.csv', index=None)
# def get_transfer(transfer_id):
#     response = requests.get('http://{}/transfer/{}'.format(orchestrator, transfer_id))
#     result = response.json()
#     data = pd.read_csv('D:/N_Prediction/data/real-time/collect/100/time.csv')
#     data = data.append([{'num_workers': int(result['num_workers']), 'time': int(result['end_time'] - result['start_time'])}], ignore_index=True)
#     data.to_csv('D:/N_Prediction/data/real-time/collect/100/time.csv', index=None)
#
# # get_transfer_init(634)
# for i in range(646,647):
#     get_transfer(i)
# get_transfer(595)



# data = pd.read_csv('D:/N_Prediction/nvme_usage_daily.csv')
# data['NVMe_written']=data['NVMe_written'].apply(lambda x: x*10000)
# data['mean']=data['mean'].apply(lambda x: x*10000)
# data.to_csv('D:/N_Prediction/nvme_usage_daily10000.csv',index=None)