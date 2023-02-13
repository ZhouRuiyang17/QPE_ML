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
# *****************************************************[1] 准备数据*****************************************************
# 读取数据集
path = 'history/230212 zs/'
x_train = np.load(path+'x_train.npy')
x_vali = np.load(path+'x_vali.npy')
x_test = np.load(path+'x_test.npy')

x_train = x_train[:,[2,3,4,5]]
x_vali = x_vali[:,[2,3,4,5]]
x_test = x_test[:,[2,3,4,5]]


y_train = np.load(path+'y_train.npy').reshape(-1,1)
y_vali = np.load(path+'y_vali.npy').reshape(-1,1)
y_test = np.load(path+'y_test.npy').reshape(-1,1)

pp_x = MinMaxScaler()
_ = pp_x.fit(x_train)
x_train = pp_x.transform(x_train)
x_vali = pp_x.transform(x_vali)
x_test = pp_x.transform(x_test)

pp_y = MinMaxScaler()
_ = pp_y.fit(y_train)
y_train = pp_y.transform(y_train)
y_vali = pp_y.transform(y_vali)
y_test = pp_y.transform(y_test)

# 张量化
x_train = torch.from_numpy(x_train.astype(np.float32))
x_vali = torch.from_numpy(x_vali.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_vali = torch.from_numpy(y_vali.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))


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


net = torch.load(path+'net_best.pt')
net.eval()

y_pred = net(x_test).detach().numpy()
y_pred = pp_y.inverse_transform(y_pred)

x_test = x_test.detach().numpy()
x_test = pp_x.inverse_transform(x_test)

y_test = y_test.detach().numpy()
y_test = pp_y.inverse_transform(y_test)

z = x_test[:,2]
zr_mayu = 0.0576*(10**(z/10))**0.557
b = 1/1.4; a = (1/300)**b
zr_300  = a     *(10**(z/10))**b
b = 1/1.6; a = (1/200)**b
zr_200  = a     *(10**(z/10))**b

def fun(x,k,b):
    return k*x+b

#%%
import pandas as pd
df = pd.DataFrame(np.zeros(shape=(5,5)))
df.iloc[:, 0] = str(len(y_test)), 'ME', 'MAE', 'RMSE', 'CC'
df.iloc[0,1:] = 'QPE_FC', 'Z=238R^1.57', 'Z=300R^1.4', 'Z=200R^1.6' 
def evaluation(t, p):
    d = (t-p)
    me = np.mean(d)
    mae = np.mean(abs(d))
    rmse = (np.mean(d**2))**0.5
    cc = np.corrcoef(t.flatten(), p.flatten())[0,1]
    return me, mae, rmse, cc
df.iloc[1:, 1] = evaluation(y_test, y_pred)
df.iloc[1:, 2] = evaluation(y_test, zr_mayu)
df.iloc[1:, 3] = evaluation(y_test, zr_300)
df.iloc[1:, 4] = evaluation(y_test, zr_200)
df.to_csv(path+'all.csv')

def plot(t, p, label=None, stat=None,num=None):
    # set rcparams
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
    z1 = z*len(z)
    
    # draw
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(x, y, label=label, c=z1, cmap='Spectral_r')
    
    ax.plot([0,100],[0,100],c='black')
    ax.set_xlabel('Observation (mm/h)')
    ax.set_ylabel('Estimation (mm/h)')
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
plot(y_test, y_pred, label='$QPE_{FC}$',num='(a)',stat=stat)
stat=[df.iloc[0,0],df.iloc[2,2],df.iloc[3,2],df.iloc[4,2]]
plot(y_test, zr_mayu, label='$Z=238R^{1.57}$',num='(b)',stat=stat)
stat=[df.iloc[0,0],df.iloc[2,3],df.iloc[3,3],df.iloc[4,3]]
plot(y_test, zr_300, label='$Z=300R^{1.4}$',num='(c)',stat=stat)
stat=[df.iloc[0,0],df.iloc[2,4],df.iloc[3,4],df.iloc[4,4]]
plot(y_test, zr_200, label='$Z=200R^{1.6}$',num='(d)',stat=stat)



limi = [0, 0.1, 10, 25, 50, np.max(y_test)+100]
for i in range(5):
    loc = np.where((y_test>=limi[i]) & (y_test<limi[i+1]))[0]
    # print(len(loc))
    
    df = pd.DataFrame(np.zeros(shape=(5,5)))
    df.iloc[:, 0] = str(len(loc)), 'ME', 'MAE', 'RMSE', 'CC'
    df.iloc[0,1:] = 'QPE_FC', 'Z=238R^1.57', 'Z=300R^1.4', 'Z=200R^1.6' 
    

    df.iloc[1:, 1] = evaluation(y_test[loc], y_pred[loc])
    df.iloc[1:, 2] = evaluation(y_test[loc], zr_mayu[loc])
    df.iloc[1:, 3] = evaluation(y_test[loc], zr_300[loc])
    df.iloc[1:, 4] = evaluation(y_test[loc], zr_200[loc])
    df.to_csv(path+str(limi[i])+'-'+str(limi[i+1])+'.csv')


# plt.scatter(z, y_test, label='$obs$')
# plt.scatter(z, y_pred, label='$QPE_{FC}$')
# plt.xlim(0,100)
# plt.ylim(0,100)
# plt.legend()
# plt.show()

# plt.scatter(z, y_test, label='$obs$')
# plt.scatter(z, zr_mayu, label='$Z=238R^{1.57}$')
# plt.xlim(0,100)
# plt.ylim(0,100)
# plt.legend()
# plt.show()

# plt.scatter(z, y_test, label='$obs$')
# plt.scatter(z, zr_300, label='$Z=300R^{1.4}$')
# plt.xlim(0,100)
# plt.ylim(0,100)
# plt.legend()
# plt.show()

# plt.scatter(z, y_test, label='$obs$')
# plt.scatter(z, zr_200, label='$Z=200R^{1.6}$')
# plt.xlim(0,100)
# plt.ylim(0,100)
# plt.legend()
# plt.show()