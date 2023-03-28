import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler, QuantileTransformer
from scipy.stats import kstest
from scipy import stats
from scipy.optimize import curve_fit
from tools import *


rain_event = np.load('data_zs_190728_190731.npy')
loc = np.where(rain_event[:, -1]<0)
rain_event[loc, -1] = 0

path = 'result/'
x_train = np.load(path+'x_train.npy')[:,[2,3,4,5]]
pp_x = MinMaxScaler()
_ = pp_x.fit(x_train)
x = pp_x.transform(rain_event[:, 2:6])

y_train = np.load(path+'y_train.npy').reshape(-1,1)
pp_y = MinMaxScaler()
_ = pp_y.fit(y_train)
y = pp_y.transform(rain_event[:, -1].reshape(-1,1))


# 张量化
x_tensor = torch.from_numpy(x.astype(np.float32))
y_tensor = torch.from_numpy(y.astype(np.float32))



# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc = nn.Sequential(
             
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()

        )

    def forward(self, x):
        out = self.fc(x)
        return out


net = torch.load(path+'net_best.pt')# 载入模型
net.eval()# 设置为eval状态
y_pred = net(x_tensor).detach().numpy()# 应用模型+解包+numpy化
y_pred = pp_y.inverse_transform(y_pred)# 逆标准化
y_pred = y_pred.flatten()
y_t = y.flatten()

z = rain_event[:, 3] # site_num, date, 0.5, 1.5
zr_mayu = 0.0576*(10**(z/10))**0.557
b = 1/1.4; a = (1/300)**b
zr_300  = a     *(10**(z/10))**b
b = 1/1.6; a = (1/200)**b
zr_200  = a     *(10**(z/10))**b


# import pandas as pd
# df = pd.DataFrame(np.zeros(shape=(5,5)))
# df.iloc[:, 0] = str(len(y_t)), 'ME', 'MAE', 'RMSE', 'CC'
# df.iloc[0,1:] = 'QPE_FC', 'Z=238R^1.57', 'Z=300R^1.4', 'Z=200R^1.6' 
# def evaluation(t, p):
#     d = (t-p)
#     me = np.mean(d)
#     mae = np.mean(abs(d))
#     rmse = (np.mean(d**2))**0.5
#     cc = np.corrcoef(t.flatten(), p.flatten())[0,1]
#     return me, mae, rmse, cc
# df.iloc[1:, 1] = evaluation(y_t, y_pred)
# df.iloc[1:, 2] = evaluation(y_t, zr_mayu)
# df.iloc[1:, 3] = evaluation(y_t, zr_300)
# df.iloc[1:, 4] = evaluation(y_t, zr_200)
# df.to_csv(path+'all_eval.csv')

# def plot(t, p, label=None, stat=None,num=None):
#     # set rcparams
#     plt.rcParams['figure.figsize']=(10,8)
#     plt.rcParams['figure.dpi']=400
#     plt.rcParams['font.size']=20
#     plt.rcParams['font.serif']='Times New Roman'
    
    
#     x = t.flatten()
#     y = p.flatten()
#     xy = np.vstack([x,y])
#     z = gaussian_kde(xy)(xy)
#     idx = z.argsort()
#     x, y, z = x[idx], y[idx], z[idx]
#     z1 = z*len(z)
    
#     # draw
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     scatter = ax.scatter(x, y, label=label, c=z1, cmap='Spectral_r')
    
#     ax.plot([0,100],[0,100],c='black')
#     ax.set_xlabel('Observation (mm/h)')
#     ax.set_ylabel('Estimation (mm/h)')
#     ax.set_ylim(0,100)
#     ax.set_xlim(0,100)
#     plt.legend(loc='upper right')
#     plt.grid()
    
#     ax.text(1,95,num)

#     if stat:
#         ax.text(1,90,'points='+str(stat[0]))
#         ax.text(1,85,'MAE='+str(stat[1])[:5])
#         ax.text(1,80,'RMSE='+str(stat[2])[:5])
#         ax.text(1,75,'CORR='+str(stat[3])[:5])
#     fig.colorbar(scatter)
#     plt.show()
# stat=[df.iloc[0,0],df.iloc[2,1],df.iloc[3,1],df.iloc[4,1]]
# plot(y_t, y_pred, label='$QPE_{FC}$',num='(a)',stat=stat)
# stat=[df.iloc[0,0],df.iloc[2,2],df.iloc[3,2],df.iloc[4,2]]
# plot(y_t, zr_mayu, label='$Z=238R^{1.57}$',num='(b)',stat=stat)
# stat=[df.iloc[0,0],df.iloc[2,3],df.iloc[3,3],df.iloc[4,3]]
# plot(y_t, zr_300, label='$Z=300R^{1.4}$',num='(c)',stat=stat)
# stat=[df.iloc[0,0],df.iloc[2,4],df.iloc[3,4],df.iloc[4,4]]
# plot(y_t, zr_200, label='$Z=200R^{1.6}$',num='(d)',stat=stat)



# limi = [0, 0.1, 10, 25, 50, np.max(y_t)+100]
# for i in range(5):
#     loc = np.where((y_t>=limi[i]) & (y_t<limi[i+1]))[0]
#     # print(len(loc))
    
#     df = pd.DataFrame(np.zeros(shape=(5,5)))
#     df.iloc[:, 0] = str(len(loc)), 'ME', 'MAE', 'RMSE', 'CC'
#     df.iloc[0,1:] = 'QPE_FC', 'Z=238R^1.57', 'Z=300R^1.4', 'Z=200R^1.6' 
    

#     df.iloc[1:, 1] = evaluation(y_t[loc], y_pred[loc])
#     df.iloc[1:, 2] = evaluation(y_t[loc], zr_mayu[loc])
#     df.iloc[1:, 3] = evaluation(y_t[loc], zr_300[loc])
#     df.iloc[1:, 4] = evaluation(y_t[loc], zr_200[loc])
#     df.to_csv(path+str(limi[i])+'-'+str(limi[i+1])+'_eval.csv')

# loc = np.where(rain_event[:,0] == 54594)
# x = rain_event[loc, 1].reshape(-1)
# y0 = rain_event[loc, -1].reshape(-1)
# y1 = y_pred[loc].reshape(-1)
# y2 = zr_mayu[loc]
# y3 = zr_300[loc]
# y4 = zr_200[loc]
# plt.plot(x[:300], y0[:300], label = '0')
# plt.plot(x[:300], y1[:300], label = '1')
# plt.plot(x[:300], y2[:300], label = '2')
# plt.plot(x[:300], y3[:300], label = '3')
# plt.plot(x[:300], y4[:300], label = '4')
# plt.legend()


#%% 统计分析
import pandas as pd
# ----所有
df = pd.DataFrame(np.zeros(shape=(5,5)))
df.iloc[:, 0] = str(len(y_t)), 'ME', 'MAE', 'RMSE', 'CC'
df.iloc[0,1:] = 'QPE_FC', 'Z=238R^1.57', 'Z=300R^1.4', 'Z=200R^1.6' 
def evaluation(t, p):
    d = (t-p)
    me = np.mean(d)
    mae = np.mean(abs(d))
    rmse = (np.mean(d**2))**0.5
    cc = np.corrcoef(t.flatten(), p.flatten())[0,1]
    return me, mae, rmse, cc
df.iloc[1:, 1] = evaluation(y_t, y_pred)
df.iloc[1:, 2] = evaluation(y_t, zr_mayu)
df.iloc[1:, 3] = evaluation(y_t, zr_300)
df.iloc[1:, 4] = evaluation(y_t, zr_200)
df.to_csv(path+'all.csv')


limi = [0, 0.1, 10, 25, 50, np.max(y_t)+100]
for i in range(5):
    loc = np.where((y_t>=limi[i]) & (y_t<limi[i+1]))[0]
    # print(len(loc))
    
    df1 = pd.DataFrame(np.zeros(shape=(5,5)))
    df1.iloc[:, 0] = str(len(loc)), 'ME', 'MAE', 'RMSE', 'CC'
    df1.iloc[0,1:] = 'QPE_FC', 'Z=238R^1.57', 'Z=300R^1.4', 'Z=200R^1.6' 
    

    df1.iloc[1:, 1] = evaluation(y_t[loc], y_pred[loc])
    df1.iloc[1:, 2] = evaluation(y_t[loc], zr_mayu[loc])
    df1.iloc[1:, 3] = evaluation(y_t[loc], zr_300[loc])
    df1.iloc[1:, 4] = evaluation(y_t[loc], zr_200[loc])
    df1.to_csv(path+str(limi[i])+'-'+str(limi[i+1])+'.csv')
#%% 作图分析

# maxi = np.max([np.max(y_t), np.max(y_pred), np.max(zr_mayu), np.max(zr_300), np.max(zr_200)])
maxi = 200
def plot(t, p, label=None, stat=None,num=None): 
    # ----set rcparams
    plt.rcParams['figure.figsize']=(10,8)
    plt.rcParams['figure.dpi']=400
    plt.rcParams['font.size']=20
    plt.rcParams['font.serif']='Times New Roman'
    
    
    x = t.flatten()
    y = p.flatten()
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    z1 = z*len(z) # 占比*总数 = 个数
    
    # ----draw
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(x, y, label=label, c=z1, cmap='Spectral_r')
    
    ax.plot([0,maxi],[0,maxi],c='black')
    ax.set_xlabel('Observation (mm/h)')
    ax.set_ylabel('Estimation (mm/h)')
    # ax.set_xlabel('Z (dBZ)')
    # ax.set_ylabel('R (mm/h)')
    ax.set_ylim(0,100)
    ax.set_xlim(0,100)
    plt.legend(loc='upper right')
    plt.grid()
    
    ax.text(1,95,num)

    if stat:
        ax.text(1,90,'points='+str(stat[0]))
        ax.text(1,85,'MAE='+str(stat[1])[:5])
        ax.text(1,80,'RMSE='+str(stat[2])[:5])
        ax.text(1,75,'CORR='+str(stat[3])[:5])
    fig.colorbar(scatter)
    plt.show()

stat=[df.iloc[0,0],df.iloc[2,1],df.iloc[3,1],df.iloc[4,1]]
plot(y_t, y_pred, label='$QPE_{FC}$',num='(a)',stat=stat)
stat=[df.iloc[0,0],df.iloc[2,2],df.iloc[3,2],df.iloc[4,2]]
plot(y_t, zr_mayu, label='$Z=238R^{1.57}$',num='(b)',stat=stat)
stat=[df.iloc[0,0],df.iloc[2,3],df.iloc[3,3],df.iloc[4,3]]
plot(y_t, zr_300, label='$Z=300R^{1.4}$',num='(c)',stat=stat)
stat=[df.iloc[0,0],df.iloc[2,4],df.iloc[3,4],df.iloc[4,4]]
plot(y_t, zr_200, label='$Z=200R^{1.6}$',num='(d)',stat=stat)


def plot2(z, r, label=None, stat=None,num=None):
    # ----set rcparams
    plt.rcParams['figure.figsize']=(10,8)
    plt.rcParams['figure.dpi']=400
    plt.rcParams['font.size']=20
    plt.rcParams['font.serif']='Times New Roman'
    
    
    x = z.flatten()
    y = r.flatten()
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    z1 = z*len(z) # 占比*总数 = 个数
    
    # ----draw
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(x, y, label=label, c=z1, cmap='Spectral_r')

    ax.set_xlabel('Z (dBZ)')
    ax.set_ylabel('R (mm/h)')
    ax.set_ylim(0,200)
    ax.set_xlim(0,80)
    plt.legend(loc='upper right')
    plt.grid()
    
    ax.text(1,185,num)

    # if stat:
    #     ax.text(1,90,'points='+str(stat[0]))
    #     ax.text(1,85,'MAE='+str(stat[1])[:5])
    #     ax.text(1,80,'RMSE='+str(stat[2])[:5])
    #     ax.text(1,75,'CORR='+str(stat[3])[:5])
    fig.colorbar(scatter)
    plt.show()

plot2(x[:,1], y_t, label='obs',num='(a)')
plot2(x[:,1], y_pred, label='$QPE_{FC}$',num='(b)')
plot2(x[:,1], zr_mayu, label='$Z=238R^{1.57}$',num='(c)')
plot2(x[:,1], zr_300, label='$Z=300R^{1.4}$',num='(d)')
plot2(x[:,1], zr_200, label='$Z=200R^{1.6}$',num='(e)')

plt.boxplot([(y_pred - y_t),(zr_mayu - y_t),(zr_300 - y_t),(zr_200 - y_t)], 
            showfliers=True, showmeans = True, meanline = True)
plt.grid()
plt.show()