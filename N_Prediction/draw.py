import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

draw rmse
datarsme = pd.read_csv('D:/N_Prediction/data/real-time/feature/rmset.csv')
data= [datarsme['rmse'].iloc[0],datarsme['rmse'].iloc[1],datarsme['rmse'].iloc[2],datarsme['rmse'].iloc[3],
                  datarsme['rmse'].iloc[4],datarsme['rmse'].iloc[5],datarsme['rmse'].iloc[6],datarsme['rmse'].iloc[7],
                  datarsme['rmse'].iloc[8],datarsme['rmse'].iloc[9],datarsme['rmse'].iloc[10]]
tick_label = ["8","10", "20", "30", "40", "50", "60", "70", "80", "90","100"]
index = np.arange(len(tick_label))
bar_width = 0.4
for a,b in zip(index,data):
 plt.text(a,b, str('{:.2f}'.format(b)), ha='center',va='bottom',fontsize=7)
plt.bar(index, data, bar_width, color="g", align="center",tick_label = tick_label)
plt.legend(['rmse'])
plt.xlabel("Num_workers")
plt.ylabel("Real-time rmse")
plt.title("Real-time RMSE")
plt.savefig('D:/N_Prediction/data/compare/picture/all_rmse.jpg',dpi = 600)
plt.show()


# draw predictiontime
# timeusedata = pd.read_csv('D:/N_Prediction/predictiontime.csv')
# firsttimeuse = timeusedata['firsttime'].mean()
# secondtimeuse = timeusedata['secondtime'].mean()
# totaltime = firsttimeuse+secondtimeuse
# timedata = [firsttimeuse,secondtimeuse,totaltime]
# bar_width = 0.25
# tick_label = ["firststep", "secondstep","total time"]
# index=np.arange(len(tick_label))
# for a,b in zip(index,timedata):   #柱子上的数字显示
#  plt.text(a,b,'%.2f'%b,ha='center',va='bottom',fontsize=10)
# plt.bar(index, timedata, bar_width, color="b", align="center", label="predicion", alpha=0.5)
#
# plt.xlabel("Prediction process")
# plt.ylabel("Prediction time use(s)")
#
# plt.xticks(index, tick_label)
# plt.title("")
# plt.legend(labelspacing=1)
# plt.savefig('D:/N_Prediction/data/compare/picture/predition time.eps',dpi = 300)
# plt.show()

#draw time compare
# datastatic = pd.read_csv('D:/N_Prediction/data/compare/single/pack/time.csv')
# datastatictime = [datastatic['time'].iloc[0],datastatic['time'].iloc[1],datastatic['time'].iloc[2],datastatic['time'].iloc[3],
#                   datastatic['time'].iloc[4],datastatic['time'].iloc[5],datastatic['time'].iloc[6],datastatic['time'].iloc[7],
#                   datastatic['time'].iloc[8],datastatic['time'].iloc[9],datastatic['time'].iloc[10]]
#
# datadynamic8 = pd.read_csv('D:/N_Prediction/data/real-time/collect/8/time.csv')
# datadynamic10 = pd.read_csv('D:/N_Prediction/data/real-time/collect/10/time.csv')
# datadynamic20 = pd.read_csv('D:/N_Prediction/data/real-time/collect/20/time.csv')
# datadynamic30 = pd.read_csv('D:/N_Prediction/data/real-time/collect/30/time.csv')
# datadynamic40 = pd.read_csv('D:/N_Prediction/data/real-time/collect/40/time.csv')
# datadynamic50 = pd.read_csv('D:/N_Prediction/data/real-time/collect/50/time.csv')
# datadynamic60 = pd.read_csv('D:/N_Prediction/data/real-time/collect/60/time.csv')
# datadynamic70 = pd.read_csv('D:/N_Prediction/data/real-time/collect/70/time.csv')
# datadynamic80 = pd.read_csv('D:/N_Prediction/data/real-time/collect/80/time.csv')
# datadynamic90 = pd.read_csv('D:/N_Prediction/data/real-time/collect/90/time.csv')
# datadynamic100 = pd.read_csv('D:/N_Prediction/data/real-time/collect/100/time.csv')
#
# datadynamictime = [datadynamic8['time'].mean(),datadynamic10['time'].mean(),datadynamic20['time'].mean(),datadynamic30['time'].mean(),
#                    datadynamic40['time'].mean(),datadynamic50['time'].mean(),datadynamic60['time'].mean(),datadynamic70['time'].mean(),
#                    datadynamic80['time'].mean(),datadynamic90['time'].mean(),datadynamic100['time'].mean()]
#
# tick_label = ["8","10", "20", "30", "40", "50", "60", "70", "80", "90","100"]
# index = np.arange(len(tick_label))
# bar_width = 0.4
# for a,b in zip(index,datastatictime):
#  plt.text(a,b, str('{:.0f}'.format(b)), ha='center',va='bottom',fontsize=7)
# plt.bar(index, datastatictime, bar_width, color="c", align="center",tick_label = tick_label)
# for a,b in zip(index+bar_width,datadynamictime):
#  plt.text(a,b, str('{:.0f}'.format(b)), ha='center',va='bottom',fontsize=7)
# plt.bar(index+bar_width, datadynamictime, bar_width, color="b", align="center",tick_label = tick_label)
# plt.legend(['static','dynamic'])
# plt.xlabel("Num_work")
# plt.ylabel("Transfer time(s)")
# plt.title("Compare transfer time with static and dynamic")
# plt.savefig('D:/N_Prediction/data/compare/picture/comparetime.jpg',dpi = 600)
# plt.show()

# draw packet loss
# datar1 = pd.read_csv('D:/N_Prediction/data/compare/single/pack/10_625.csv')
# datar2 = pd.read_csv('D:/N_Prediction/data/compare/single/pack/20_626.csv')
# datar3 = pd.read_csv('D:/N_Prediction/data/compare/single/pack/30_627.csv')
# datar4 = pd.read_csv('D:/N_Prediction/data/compare/single/pack/40_628.csv')
# datar5 = pd.read_csv('D:/N_Prediction/data/compare/single/pack/50_629.csv')
# datar6 = pd.read_csv('D:/N_Prediction/data/compare/single/pack/60_630.csv')
# datar7 = pd.read_csv('D:/N_Prediction/data/compare/single/pack/70_631.csv')
# datar8 = pd.read_csv('D:/N_Prediction/data/compare/single/pack/80_632.csv')
# datar9 = pd.read_csv('D:/N_Prediction/data/compare/single/pack/90_633.csv')
# datar10 = pd.read_csv('D:/N_Prediction/data/compare/single/pack/100_634.csv')
#
#
# pack1 = datar1['Packet_losses'].mean()
# pack2 = datar2['Packet_losses'].mean()
# pack3 = datar3['Packet_losses'].mean()
# pack4 = datar4['Packet_losses'].mean()
# pack5 = datar5['Packet_losses'].mean()
# pack6 = datar6['Packet_losses'].mean()
# pack7 = datar7['Packet_losses'].mean()
# pack8 = datar8['Packet_losses'].mean()
# pack9 = datar9['Packet_losses'].mean()
# pack10 = datar10['Packet_losses'].mean()
#
#
# data1 = pd.read_csv('D:/N_Prediction/10/p/meanallp.csv')
# data2 = pd.read_csv('D:/N_Prediction/20/p/meanallp.csv')
# data3 = pd.read_csv('D:/N_Prediction/30/p/meanallp.csv')
# data4 = pd.read_csv('D:/N_Prediction/40/p/meanallp.csv')
# data5 = pd.read_csv('D:/N_Prediction/50/p/meanallp.csv')
# data6 = pd.read_csv('D:/N_Prediction/60/p/meanallp.csv')
# data7 = pd.read_csv('D:/N_Prediction/70/p/meanallp.csv')
# data8 = pd.read_csv('D:/N_Prediction/80/p/meanallp.csv')
# data9 = pd.read_csv('D:/N_Prediction/90/p/meanallp.csv')
# p1 = data1['Packet_losses'].mean()
# p2 = data2['Packet_losses'].mean()
# p3 = data3['Packet_losses'].mean()
# p4 = data4['Packet_losses'].mean()
# p5 = data5['Packet_losses'].mean()
# p6 = data6['Packet_losses'].mean()
# p7 = data7['Packet_losses'].mean()
# p8 = data8['Packet_losses'].mean()
# p9 = data9['Packet_losses'].mean()
#
# data = [pack1, pack2, pack3, pack4, pack5, pack6, pack7, pack8, pack9]
# datap = [p1,p2,p3,p4,p5,p6,p7,p8,p9]
# tick_label = ["10", "20", "30", "40", "50", "60", "70", "80", "90"]
# index = np.arange(len(tick_label))
# bar_width = 0.25
# plt.bar(index, data, bar_width, color="b", align="center",tick_label = tick_label)
# plt.bar(index+bar_width, datap, bar_width, color="g", align="center",tick_label = tick_label)
# plt.legend(['no_prediction','prediction'])
# plt.xlabel("Num_work")
# plt.ylabel("Packet_losses")
# plt.title("Compare Packet Loss")
# plt.savefig('D:/N_Prediction/data/compare/picture/comparepacketloss.eps',dpi = 600)
# plt.show()
