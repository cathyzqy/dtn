import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

#get all_num rmse
pathsave = 'D:/N_Prediction/data/real-time/feature/rmset.csv'
data1 = pd.read_csv('D:/N_Prediction/data/real-time/feature/8/rmse.csv')
data2 = pd.read_csv('D:/N_Prediction/data/real-time/feature/10/rmse.csv')
data3 = pd.read_csv('D:/N_Prediction/data/real-time/feature/20/rmse.csv')
data4 = pd.read_csv('D:/N_Prediction/data/real-time/feature/30/rmse.csv')
data5 = pd.read_csv('D:/N_Prediction/data/real-time/feature/40/rmse.csv')
data6 = pd.read_csv('D:/N_Prediction/data/real-time/feature/50/rmse.csv')
data7 = pd.read_csv('D:/N_Prediction/data/real-time/feature/60/rmse.csv')
data8 = pd.read_csv('D:/N_Prediction/data/real-time/feature/70/rmse.csv')
data9 = pd.read_csv('D:/N_Prediction/data/real-time/feature/80/rmse.csv')
data10 = pd.read_csv('D:/N_Prediction/data/real-time/feature/90/rmse.csv')
data11 = pd.read_csv('D:/N_Prediction/data/real-time/feature/100/rmse.csv')


datarsme = pd.DataFrame(columns=['num_workers', 'rmse'])
datarsme = datarsme.append([{'num_workers': data1['num_workers'][0], 'rmse': data1['rmse'].mean()}], ignore_index=True)
datarsme = datarsme.append([{'num_workers': data2['num_workers'][0], 'rmse': data2['rmse'].mean()}], ignore_index=True)
datarsme = datarsme.append([{'num_workers': data3['num_workers'][0], 'rmse': data3['rmse'].mean()}], ignore_index=True)
datarsme = datarsme.append([{'num_workers': data4['num_workers'][0], 'rmse': data4['rmse'].mean()}], ignore_index=True)
datarsme = datarsme.append([{'num_workers': data5['num_workers'][0], 'rmse': data5['rmse'].mean()}], ignore_index=True)
datarsme = datarsme.append([{'num_workers': data6['num_workers'][0], 'rmse': data6['rmse'].mean()}], ignore_index=True)
datarsme = datarsme.append([{'num_workers': data7['num_workers'][0], 'rmse': data7['rmse'].mean()}], ignore_index=True)
datarsme = datarsme.append([{'num_workers': data8['num_workers'][0], 'rmse': data8['rmse'].mean()}], ignore_index=True)
datarsme = datarsme.append([{'num_workers': data9['num_workers'][0], 'rmse': data9['rmse'].mean()}], ignore_index=True)
datarsme = datarsme.append([{'num_workers': data10['num_workers'][0], 'rmse': data10['rmse'].mean()}], ignore_index=True)
datarsme = datarsme.append([{'num_workers': data11['num_workers'][0], 'rmse': data11['rmse'].mean()}], ignore_index=True)
datarsme.to_csv(pathsave,index=None)



# get transfer mean
# path1 = 'D:/N_Prediction/60/p/'
# pathsave = 'D:/N_Prediction/60/p/meanall.csv'
# files = os.listdir(path1)
# for file in files:
#     data = pd.read_csv(path1 + file)
#     if not os.path.isfile(pathsave):
#         datamean = pd.DataFrame(columns=['num_workers', 'CPU','Memory_used','NVMe_from_ceph','NVMe_from_transfer','NVMe_total_util','Goodput','Packet_losses','network_throughput'],
#                             data=[[data['num_workers'].mean(), data['CPU'].mean(), data['Memory_used'].mean(),data['NVMe_from_ceph'].mean(),data['NVMe_from_transfer'].mean(),data['NVMe_total_util'].mean(),data['Goodput'].mean(),data['Packet_losses'].mean(),data['network_throughput'].mean()]])
#     else:
#         datamean = pd.read_csv(pathsave)
#         datamean = datamean.append([{'num_workers': data['num_workers'].mean(), 'CPU': data['CPU'].mean(), 'Memory_used' : data['Memory_used'].mean(), 'NVMe_from_ceph': data['NVMe_from_ceph'].mean(), 'NVMe_from_transfer': data['NVMe_from_transfer'].mean(),'NVMe_total_util': data['NVMe_total_util'].mean(),'Goodput':  data['Goodput'].mean(),'Packet_losses': data['Packet_losses'].mean(),'network_throughput': data['network_throughput'].mean()}], ignore_index=True)
#
#
#     datamean.to_csv(pathsave, index=None)


#NVMe influence
# data8raw = pd.read_csv('D:/N_Prediction/data/compare/single/8.csv')
# data20raw = pd.read_csv('D:/N_Prediction/data/compare/single/20.csv')
# data30raw = pd.read_csv('D:/N_Prediction/data/compare/single/30.csv')
# data40raw = pd.read_csv('D:/N_Prediction/data/compare/single/40.csv')
# data50raw = pd.read_csv('D:/N_Prediction/data/compare/single/50.csv')
# data60raw = pd.read_csv('D:/N_Prediction/data/compare/single/60.csv')
# data70raw = pd.read_csv('D:/N_Prediction/data/compare/single/70.csv')
# data80raw = pd.read_csv('D:/N_Prediction/data/compare/single/80.csv')
# data90raw = pd.read_csv('D:/N_Prediction/data/compare/single/90.csv')
#
# data8real = pd.read_csv('D:/N_Prediction/8/meanall.csv')
# data20real = pd.read_csv('D:/N_Prediction/20/meanall.csv')
# data30real = pd.read_csv('D:/N_Prediction/30/meanall.csv')
# data40real = pd.read_csv('D:/N_Prediction/40/meanall.csv')
# data50real = pd.read_csv('D:/N_Prediction/50/meanall.csv')
# data60real = pd.read_csv('D:/N_Prediction/60/meanall.csv')
# data70real = pd.read_csv('D:/N_Prediction/70/meanall.csv')
# data80real = pd.read_csv('D:/N_Prediction/80/meanall.csv')
# data90real = pd.read_csv('D:/N_Prediction/90/meanall.csv')


# NVMe_from_cephraw8 = data8raw['NVMe_from_ceph'].mean()
# NVMe_from_transferraw8 = data8raw['NVMe_from_transfer'].mean()
# NVMe_total_utilraw8 = data8raw['NVMe_total_util'].mean()

# NVMe_from_cephraw20 = data20raw['NVMe_from_ceph'].mean()
# NVMe_from_transferraw20 = data20raw['NVMe_from_transfer'].mean()
# NVMe_total_utilraw20 = data20raw['NVMe_total_util'].mean()

# NVMe_from_cephraw30 = data30raw['NVMe_from_ceph'].mean()
# NVMe_from_transferraw30 = data30raw['NVMe_from_transfer'].mean()
# NVMe_total_utilraw30 = data30raw['NVMe_total_util'].mean()

# NVMe_from_cephraw40 = data40raw['NVMe_from_ceph'].mean()
# NVMe_from_transferraw40 = data40raw['NVMe_from_transfer'].mean()
# NVMe_total_utilraw40 = data40raw['NVMe_total_util'].mean()

# NVMe_from_cephraw50 = data50raw['NVMe_from_ceph'].mean()
# NVMe_from_transferraw50 = data50raw['NVMe_from_transfer'].mean()
# NVMe_total_utilraw50 = data50raw['NVMe_total_util'].mean()

# NVMe_from_cephraw60 = data60raw['NVMe_from_ceph'].mean()
# NVMe_from_transferraw60 = data60raw['NVMe_from_transfer'].mean()
# NVMe_total_utilraw60 = data60raw['NVMe_total_util'].mean()

# NVMe_from_cephraw70 = data70raw['NVMe_from_ceph'].mean()
# NVMe_from_transferraw70 = data70raw['NVMe_from_transfer'].mean()
# NVMe_total_utilraw70 = data70raw['NVMe_total_util'].mean()

# NVMe_from_cephraw80 = data80raw['NVMe_from_ceph'].mean()
# NVMe_from_transferraw80 = data80raw['NVMe_from_transfer'].mean()
# NVMe_total_utilraw80 = data80raw['NVMe_total_util'].mean()

# NVMe_from_cephraw90 = data90raw['NVMe_from_ceph'].mean()
# NVMe_from_transferraw90 = data90raw['NVMe_from_transfer'].mean()
# NVMe_total_utilraw90 = data90raw['NVMe_total_util'].mean()

# NVMe_from_ceph8 = data8real['NVMe_from_ceph'].mean()
# NVMe_from_transfer8= data8real['NVMe_from_transfer'].mean()
# NVMe_total_util8 = data8real['NVMe_total_util'].mean()

# NVMe_from_ceph20 = data20real['NVMe_from_ceph'].mean()
# NVMe_from_transfer20= data20real['NVMe_from_transfer'].mean()
# NVMe_total_util20 = data20real['NVMe_total_util'].mean()

# NVMe_from_ceph30 = data30real['NVMe_from_ceph'].mean()
# NVMe_from_transfer30= data30real['NVMe_from_transfer'].mean()
# NVMe_total_util30 = data30real['NVMe_total_util'].mean()

# NVMe_from_ceph40 = data40real['NVMe_from_ceph'].mean()
# NVMe_from_transfer40= data40real['NVMe_from_transfer'].mean()
# NVMe_total_util40 = data40real['NVMe_total_util'].mean()

# NVMe_from_ceph50 = data50real['NVMe_from_ceph'].mean()
# NVMe_from_transfer50= data50real['NVMe_from_transfer'].mean()
# NVMe_total_util50 = data50real['NVMe_total_util'].mean()

# NVMe_from_ceph60 = data60real['NVMe_from_ceph'].mean()
# NVMe_from_transfer60= data60real['NVMe_from_transfer'].mean()
# NVMe_total_util60 = data60real['NVMe_total_util'].mean()

# NVMe_from_ceph70 = data70real['NVMe_from_ceph'].mean()
# NVMe_from_transfer70= data70real['NVMe_from_transfer'].mean()
# NVMe_total_util70 = data70real['NVMe_total_util'].mean()

# NVMe_from_ceph80 = data80real['NVMe_from_ceph'].mean()
# NVMe_from_transfer80= data80real['NVMe_from_transfer'].mean()
# NVMe_total_util80 = data80real['NVMe_total_util'].mean()

# NVMe_from_ceph90 = data90real['NVMe_from_ceph'].mean()
# NVMe_from_transfer90= data90real['NVMe_from_transfer'].mean()
# NVMe_total_util90 = data90real['NVMe_total_util'].mean()


# raw = [NVMe_from_cephraw8,NVMe_from_cephraw20,NVMe_from_cephraw30,NVMe_from_cephraw40,NVMe_from_cephraw50,NVMe_from_cephraw60,NVMe_from_cephraw70,NVMe_from_cephraw80,NVMe_from_cephraw90]
# real = [NVMe_from_ceph8,NVMe_from_ceph20,NVMe_from_ceph30,NVMe_from_ceph40,NVMe_from_ceph50,NVMe_from_ceph60,NVMe_from_ceph70,NVMe_from_ceph80,NVMe_from_ceph90]
# raw = [NVMe_from_transferraw8,NVMe_from_transferraw20,NVMe_from_transferraw30,NVMe_from_transferraw40,NVMe_from_transferraw50,NVMe_from_transferraw60,NVMe_from_transferraw70,NVMe_from_transferraw80,NVMe_from_transferraw90]
# real =[NVMe_from_transfer8,NVMe_from_transfer20,NVMe_from_transfer30,NVMe_from_transfer40,NVMe_from_transfer50,NVMe_from_transfer60,NVMe_from_transfer70,NVMe_from_transfer80,NVMe_from_transfer90]
# bar_width = 0.25
# tick_label = ["8", "20", "30", "40", "50","60","70","80","90"]
# index=np.arange(len(tick_label))
# for a,b in zip(index,raw):   #柱子上的数字显示
#  plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=10)
# plt.bar(index, raw, bar_width, align="center", color="c", label="no_prediction")
# plt.bar(index,timedata,bar_width,color='steelblue',align="center",tick_label = tick_label,label='improve')
# for a,b in zip(index,real):   #柱子上的数字显示
#  plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=10)
# plt.bar(index+bar_width, real, bar_width, color="b", align="center", label="predicion", alpha=0.5)

# plt.xlabel('')
# plt.ylabel("NVMe_from_transfer")

# plt.xticks(index+bar_width/2, tick_label)
# plt.xticks(index, tick_label)
# plt.title("")
# plt.legend(labelspacing=1)
# plt.savefig('D:/N_Prediction/data/compare/picture/NVMe_from_transfer.jpg',dpi = 300)
# plt.show()




# datatimenvme10 = pd.read_csv('D:/N_Prediction/8/nvme10/time.csv')
# datatimenvme20 = pd.read_csv('D:/N_Prediction/8/nvme20/time.csv')
# datatimenvme30 = pd.read_csv('D:/N_Prediction/8/nvme30/time.csv')
#
# datatimenvme10p = pd.read_csv('D:/N_Prediction/8/nvme10/timep.csv')
# datatimenvme20p = pd.read_csv('D:/N_Prediction/8/nvme20/timep.csv')
# datatimenvme30p = pd.read_csv('D:/N_Prediction/8/nvme30/timep.csv')
#
# datanvme10 = pd.read_csv('D:/N_Prediction/8/nvme10/602.csv')
# datanvme20 = pd.read_csv('D:/N_Prediction/8/nvme20/597.csv')
# datanvme30 = pd.read_csv('D:/N_Prediction/8/nvme30/596.csv')
#
# datanvme10p = pd.read_csv('D:/N_Prediction/8/nvme10/601p.csv')
# datanvme20p = pd.read_csv('D:/N_Prediction/8/nvme20/599p.csv')
# datanvme30p = pd.read_csv('D:/N_Prediction/8/nvme30/598p.csv')

# datatimenvme10 = datanvme10['time'].iloc[0,0] - datanvme10['time'].iloc[-1,0]
# datatimenvme20 = datanvme20['time'].iloc[0,0] - datanvme20['time'].iloc[-1,0]
# datatimenvme30 = datanvme20['time'].iloc[0,0] - datanvme30['time'].iloc[-1,0]
#
# datatimenvme10p = datanvme10p['time'].iloc[0,0] - datanvme10p['time'].iloc[-1,0]
# datatimenvme20p = datanvme20p['time'].iloc[0,0] - datanvme20p['time'].iloc[-1,0]
# datatimenvme30p = datanvme20p['time'].iloc[0,0] - datanvme30p['time'].iloc[-1,0]

# time10 = (datatimenvme10['time'].mean()-datatimenvme10p['time'].mean()) / datatimenvme10['time'].mean()
# cpu10 = (datanvme10['CPU'].mean() - datanvme10p['CPU'].mean()) / datanvme10['CPU'].mean()
# throughput10 = (datanvme10['Goodput'].mean() - datanvme10p['Goodput'].mean()) / datanvme10['Goodput'].mean()
# data10 = [time10*100]
#
# time20 = (datatimenvme20['time'].mean()-datatimenvme20p['time'].mean()) / datatimenvme20['time'].mean()
# cpu20 = (datanvme20['CPU'].mean() - datanvme20p['CPU'].mean()) / datanvme20['CPU'].mean()
# throughput20 = (datanvme20['Goodput'].mean() - datanvme20p['Goodput'].mean()) / datanvme20['Goodput'].mean()
# data20 = [time20*100]
#
# time30 = (datatimenvme30['time'].mean()-datatimenvme30p['time'].mean()) / datatimenvme30['time'].mean()
# cpu30 = (datanvme30['CPU'].mean() - datanvme30p['CPU'].mean()) / datanvme30['CPU'].mean()
# throughput30 = (datanvme30['Goodput'].mean() - datanvme30p['Goodput'].mean()) / datanvme30['Goodput'].mean()
# data30 = [time30*100]
#
# data = [time10*100,time20*100,time30*100]
# # tick_label = ["nvme10","nvme20", "nvme30"]
# # index = np.arange(len(tick_label))
#
# def to_percent(temp, position):
#  return '%1.0f' % (1 * temp) + '%'
#
# plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
#
# bar_width = 0.1
# plt.bar(index, data10, bar_width, color="b", align="center",tick_label = tick_label)
# for a,b in zip(index,data10):
#  plt.text(a,b, str('{:.2f}'.format(b)) +'%', ha='center',va='bottom',fontsize=7) #位置，高度，内容，居中

# plt.bar(index, data, bar_width, color="g", align="center",tick_label = tick_label)
# for a,b in zip(index,data):
#  plt.text(a,b, str('{:.2f}'.format(b)) +'%', ha='center',va='bottom',fontsize=7) #位置，高度，内容，居中
#
# # plt.bar(index+bar_width, data30, bar_width, color="g", align="center")
# # for a,b in zip(index+bar_width,data30):
# #  plt.text(a,b, str('{:.2f}'.format(b)) +'%', ha='center',va='bottom',fontsize=7) #位置，高度，内容，居中
#
# # plt.legend(['time reduce '])
# plt.xlabel("Multiples of NVMe")
# plt.ylabel("Percentage of time reduction")
# plt.title("The influence of the nvme")
# plt.savefig('D:/N_Prediction/data/compare/picture/nvmetiminfluence.eps',dpi = 600)
# plt.show()



