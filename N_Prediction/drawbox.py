import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# draw time reduce proportion
data1 = pd.read_csv('D:/N_Prediction/data/real-time/collect/8/time.csv')
data2 = pd.read_csv('D:/N_Prediction/data/real-time/collect/10/time.csv')
data3 = pd.read_csv('D:/N_Prediction/data/real-time/collect/20/time.csv')
data4 = pd.read_csv('D:/N_Prediction/data/real-time/collect/30/time.csv')
data5 = pd.read_csv('D:/N_Prediction/data/real-time/collect/40/time.csv')
data6 = pd.read_csv('D:/N_Prediction/data/real-time/collect/50/time.csv')
data7 = pd.read_csv('D:/N_Prediction/data/real-time/collect/60/time.csv')
data8 = pd.read_csv('D:/N_Prediction/data/real-time/collect/70/time.csv')
data9 = pd.read_csv('D:/N_Prediction/data/real-time/collect/80/time.csv')
data10 = pd.read_csv('D:/N_Prediction/data/real-time/collect/90/time.csv')
data11 = pd.read_csv('D:/N_Prediction/data/real-time/collect/100/time.csv')

data1['time'] = data1['time'].apply(lambda x: (1838-x) / 1838)
data2['time'] = data2['time'].apply(lambda x: (1756-x) / 1756)
data3['time'] = data3['time'].apply(lambda x: (1401-x) / 1401)
data4['time'] = data4['time'].apply(lambda x: (926-x) / 926)
data5['time'] = data5['time'].apply(lambda x: (770-x) / 770)
data6['time'] = data6['time'].apply(lambda x: (707-x) / 707)
data7['time'] = data7['time'].apply(lambda x: (722-x) / 722)
data8['time'] = data8['time'].apply(lambda x: (728-x) / 728)
data9['time'] = data9['time'].apply(lambda x: (736-x) / 736)
data10['time'] = data10['time'].apply(lambda x: (750-x) / 734)
data11['time'] = data11['time'].apply(lambda x: (772-x) / 734)

s1 = np.array(data1['time'])
s1 = s1.tolist()
s2 = np.array(data2['time'])
s2 = s2.tolist()
s3 = np.array(data3['time'])
s3 = s3.tolist()
s4 = np.array(data4['time'])
s4 = s4.tolist()
s5 = np.array(data5['time'])
s5 = s5.tolist()
s6 = np.array(data6['time'])
s6 = s6.tolist()
s7 = np.array(data7['time'])
s7 = s7.tolist()
s8 = np.array(data8['time'])
s8 = s8.tolist()
s9 = np.array(data9['time'])
s9 = s9.tolist()
s10 = np.array(data10['time'])
s10 = s10.tolist()
s10 = np.array(data10['time'])
s10 = s10.tolist()
s11 = np.array(data11['time'])
s11 = s11.tolist()
data = pd.DataFrame({
    "8": s1,
    "10": s2,
    "20": s3,
    "30": s4,
    "40": s5,
    "50": s6,
    "60": s7,
    "70": s8,
    "80": s9,
    "90": s10,
    "100": s11,

})
data.boxplot()
plt.ylabel("Time reduced proportion")
plt.xlabel("Begin num_workers")
plt.title('Time reduced proportion')

plt.savefig('D:/N_Prediction/data/compare/picture/allnum_proportion.eps')
plt.show()

# data1 = pd.read_csv('D:/N_Prediction/8/meanall.csv')
# data2 = pd.read_csv('D:/N_Prediction/10/p/meanall.csv')
# data3 = pd.read_csv('D:/N_Prediction/20/p/meanall.csv')
# data4 = pd.read_csv('D:/N_Prediction/30/p/meanall.csv')
# data5 = pd.read_csv('D:/N_Prediction/40/p/meanall.csv')
# data6 = pd.read_csv('D:/N_Prediction/50/p/meanall.csv')
# data7 = pd.read_csv('D:/N_Prediction/60/p/meanall.csv')
# data8 = pd.read_csv('D:/N_Prediction/70/p/meanall.csv')
# data9 = pd.read_csv('D:/N_Prediction/80/p/meanall.csv')
# data10 = pd.read_csv('D:/N_Prediction/90/p/meanall.csv')

# datar1 = pd.read_csv('D:/N_Prediction/data/historical/rawdata/nomaldata/15s/8.csv')
# datar2 = pd.read_csv('D:/N_Prediction/data/historical/rawdata/nomaldata/15s/10.csv')
# datar3 = pd.read_csv('D:/N_Prediction/data/historical/rawdata/nomaldata/15s/20.csv')
# datar4 = pd.read_csv('D:/N_Prediction/data/historical/rawdata/nomaldata/15s/30.csv')
# datar5 = pd.read_csv('D:/N_Prediction/data/historical/rawdata/nomaldata/15s/40.csv')
# datar6 = pd.read_csv('D:/N_Prediction/data/historical/rawdata/nomaldata/15s/50.csv')
# datar7 = pd.read_csv('D:/N_Prediction/data/historical/rawdata/nomaldata/15s/60.csv')
# datar8 = pd.read_csv('D:/N_Prediction/data/historical/rawdata/nomaldata/15s/70.csv')
# datar9 = pd.read_csv('D:/N_Prediction/data/historical/rawdata/nomaldata/15s/80.csv')
# datar10 = pd.read_csv('D:/N_Prediction/data/historical/rawdata/nomaldata/15s/90.csv')

# datar1 = pd.read_csv('D:/N_Prediction/data/compare/single/pack/10_237.csv')
# datar2 = pd.read_csv('D:/N_Prediction/data/compare/single/pack/20_226.csv')
# datar3 = pd.read_csv('D:/N_Prediction/data/compare/single/pack/30_322.csv')
# datar4 = pd.read_csv('D:/N_Prediction/data/compare/single/pack/40_205.csv')
# datar5 = pd.read_csv('D:/N_Prediction/data/compare/single/pack/50_195.csv')
# datar6 = pd.read_csv('D:/N_Prediction/data/compare/single/pack/60_157.csv')
# datar7 = pd.read_csv('D:/N_Prediction/data/compare/single/pack/70_98.csv')
# datar8 = pd.read_csv('D:/N_Prediction/data/compare/single/pack/80_131.csv')
# datar9 = pd.read_csv('D:/N_Prediction/data/compare/single/pack/90_100.csv')

# cpu1 = datar1['CPU'].mean()
# cpu2 = datar2['CPU'].mean()
# cpu3 = datar3['CPU'].mean()
# cpu4 = datar4['CPU'].mean()
# cpu5 = datar5['CPU'].mean()
# cpu6 = datar6['CPU'].mean()
# cpu7 = datar7['CPU'].mean()
# cpu8 = datar8['CPU'].mean()
# cpu9 = datar9['CPU'].mean()
# cpu10 = datar10['CPU'].mean()
# cpu1 = datar1['Packet_losses'].mean()
# cpu2 = datar2['Packet_losses'].mean()
# cpu3 = datar3['Packet_losses'].mean()
# cpu4 = datar4['Packet_losses'].mean()
# cpu5 = datar5['Packet_losses'].mean()
# cpu6 = datar6['Packet_losses'].mean()
# cpu7 = datar7['Packet_losses'].mean()
# cpu8 = datar8['Packet_losses'].mean()
# cpu9 = datar9['Packet_losses'].mean()

# data1['CPU'] = data1['CPU'].apply(lambda x: (cpu1-x) / cpu1)
# data2['CPU'] = data2['CPU'].apply(lambda x: (cpu2-x) / cpu2)
# data3['CPU'] = data3['CPU'].apply(lambda x: (cpu3-x) / cpu3)
# data4['CPU'] = data4['CPU'].apply(lambda x: (cpu4-x) / cpu4)
# data5['CPU'] = data5['CPU'].apply(lambda x: (cpu5-x) / cpu5)
# data6['CPU'] = data6['CPU'].apply(lambda x: (cpu6-x) / cpu6)
# data7['CPU'] = data7['CPU'].apply(lambda x: (cpu7-x) / cpu7)
# data8['CPU'] = data8['CPU'].apply(lambda x: (cpu8-x) / cpu8)
# data9['CPU'] = data9['CPU'].apply(lambda x: (cpu9-x) / cpu9)
# data10['CPU'] = data10['CPU'].apply(lambda x: (cpu10-x) / cpu10)


# s1 = np.array(data1['CPU'])
# s1 = s1.tolist()
# s2 = np.array(data2['Packet_losses'])
# s2 = s2.tolist()
# s3 = np.array(data3['Packet_losses'])
# s3 = s3.tolist()
# s4 = np.array(data4['Packet_losses'])
# s4 = s4.tolist()
# s5 = np.array(data5['Packet_losses'])
# s5 = s5.tolist()
# s6 = np.array(data6['Packet_losses'])
# s6 = s6.tolist()
# s7 = np.array(data7['Packet_losses'])
# s7 = s7.tolist()
# s8 = np.array(data8['Packet_losses'])
# s8 = s8.tolist()
# s9 = np.array(data9['Packet_losses'])
# s9 = s9.tolist()
# s10 = np.array(data10['Packet_losses'])
# s10 = s10.tolist()
# data = pd.DataFrame({
#     # "8": s1,
#     "10": s2,
#     "20": s3,
#     "30": s4,
#     "40": s5,
#     "50": s6,
#     "60": s7,
#     "70": s8,
#     "80": s9,
#     "90": s10,
#
# })
# data.boxplot()
# plt.ylabel("packet_loss")
# plt.xlabel("Begin num_workers")
# plt.savefig('D:/N_Prediction/data/compare/picture/allnum_cpu.eps')
# plt.show()

s1 = [-0.043, -0.035, -0.037]
s2 = [0.078, 0.089, 0.096]
s12 = [0.035, 0.054, 0.059]
s3 = [-0.016, -0.132, -0.035]
s4 = [0.162, 0.187, 0.178]
s34 = [0.146, 0.055, 0.143]
s5 = [-0.100, -0.140, -0.137]
s6 = [0.243, 0.214, 0.213]
s56 = [0.143, 0.074, 0.076]
data = pd.DataFrame({
    "1e4(first)": s1,
    "1e4(second)": s2,
    "1e4(total)": s12,
    "2e4(first)": s3,
    "2e4(second)": s4,
    "2e4(total)": s34,
    "3e4(first)": s5,
    "3e4(second)": s6,
    "3e4(total)": s56
})
data.boxplot()
plt.ylabel("Time reduced proportion")
plt.xlabel("Second transfer size")
plt.title("30% to 70% of workers compared to 50% to 50% of workers")
# plt.savefig('D:/N_Prediction/data/compare/picture/two transfer.jpg')
plt.show()
