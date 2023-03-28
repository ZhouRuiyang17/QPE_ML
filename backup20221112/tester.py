import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler, QuantileTransformer
from scipy.stats import kstest
from scipy import stats
# *****************************************************[1] 准备数据*****************************************************
# 读取数据集
path = 'result/'
x_train = np.load(path+'x_train.npy')[:,-1:]#.reshape(-1,1)
x_vali = np.load(path+'x_vali.npy')[:,-1:]#.reshape(-1,1)
x_test = np.load(path+'x_test.npy')[:,-1:]#.reshape(-1,1)
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
            
            # # ver 1
            # nn.Linear(3, 64),
            # # nn.Dropout(0.2),
            # nn.ReLU(),
            # nn.Linear(64, 128),
            # # nn.Dropout(0.2),
            # nn.ReLU(),
            # nn.Linear(128, 256),
            # # nn.Dropout(0.2),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # # nn.Dropout(0.2),
            # nn.ReLU(),
            # nn.Linear(128, 64),
            # # nn.Dropout(0.2),
            # nn.ReLU(),
            # nn.Linear(64, 1),
            # nn.Sigmoid()
            # # nn.ReLU()
            
            # ver 2
            nn.Linear(1, 128),
            # nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 64),
            # nn.Dropout(0.2),
            nn.ReLU(),
            # nn.Linear(50, 50),
            # # nn.Dropout(0.2),
            # nn.ReLU(),
            # nn.Linear(50, 50),
            # # nn.Dropout(0.2),
            # nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
            # nn.ReLU(),

        )

    def forward(self, x):
        out = self.fc(x)
        return out


net = torch.load(path+'net_best.pt')
net.eval()
y_pred = net(x_test).detach().numpy()
x_test = x_test.detach().numpy()
y_test = y_test.detach().numpy()
y_pred = pp_y.inverse_transform(y_pred)
x_test = pp_x.inverse_transform(x_test)
y_test = pp_y.inverse_transform(y_test)

zr_mayu = 0.0576*(10**(x_test[:,0]/10))**0.557
b = 1/1.4; a = (1/300)**b
zr_300  = a     *(10**(x_test[:,0]/10))**b
b = 1/1.6; a = (1/200)**b
zr_200  = a     *(10**(x_test[:,0]/10))**b

def plot(t, p, label=None, marker='x'):
    
    x = t.flatten()
    y = p.flatten()
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    plt.scatter(x, y, label=label, marker = marker, c=z, cmap='Spectral_r')
    
    # plt.scatter(t, p, label=label, marker = marker)
    # plt.scatter(y_test, zr_300, label='$Z=300R^{1.4}$', marker = '1', alpha = 0.5)
    # plt.scatter(y_test, zr_200, label='$Z=200R^{1.6}$', marker = 'v', alpha = 0.5)
    # plt.scatter(y_test, zr_mayu, label='$R=0.0576Z^{0.557}$', marker = 'x', alpha = 0.5)
    # plt.scatter(y_test, y_pred, label='ML', marker = '.', alpha = 0.5)
    plt.plot([0,100],[0,100])
    plt.xlabel('truth')
    plt.ylabel('pred')
    plt.ylim(0,100)
    plt.xlim(0,100)
    plt.legend(loc='upper left')
    plt.colorbar()
    plt.show()

plot(y_test, y_pred, label='ML', marker = 'x')
plot(y_test, zr_mayu, label='$R=0.0576Z^{0.557}$', marker = 'x',)
plot(y_test, zr_300, label='$Z=300R^{1.4}$', marker = 'x')
plot(y_test, zr_200, label='$Z=200R^{1.6}$', marker = 'x')
#%%
import pandas as pd
df = pd.DataFrame(np.zeros(shape=(5,5)))
df.iloc[:, 0] = '', 'ME', 'MAE', 'RMSE', 'CC'
df.iloc[0,1:] = 'ML', 'R=0.0576Z^0.557', 'Z=300R^1.4', 'Z=200R^1.6' 
def pingjia(t, p):
    d = (t-p)
    me = np.mean(d)
    mae = np.mean(abs(d))
    rmse = (np.mean(d**2))**0.5
    cc = np.corrcoef(t.flatten(), p.flatten())[0,1]
    return me, mae, rmse, cc
df.iloc[1:, 1] = pingjia(y_test, y_pred)
df.iloc[1:, 2] = pingjia(y_test, zr_mayu)
df.iloc[1:, 3] = pingjia(y_test, zr_300)
df.iloc[1:, 4] = pingjia(y_test, zr_200)
df.to_csv(path+'all.csv')

limi = [0, 0.1, 10, 25, 50, 200]
for i in range(5):
    df = pd.DataFrame(np.zeros(shape=(5,5)))
    df.iloc[:, 0] = '', 'ME', 'MAE', 'RMSE', 'CC'
    df.iloc[0,1:] = 'ML', 'R=0.0576Z^0.557', 'Z=300R^1.4', 'Z=200R^1.6' 
    
    loc = np.where((y_test>=limi[i]) & (y_test<limi[i+1]))[0]
    print(len(loc))
    df.iloc[1:, 1] = pingjia(y_test[loc], y_pred[loc])
    df.iloc[1:, 2] = pingjia(y_test[loc], zr_mayu[loc])
    df.iloc[1:, 3] = pingjia(y_test[loc], zr_300[loc])
    df.iloc[1:, 4] = pingjia(y_test[loc], zr_200[loc])
    df.to_csv(path+str(limi[i])+'-'+str(limi[i+1])+'.csv')


# fff = open(path+'eval.txt','w')
# for i in range(5):
#     loc = np.where((y_test>=limi[i]) & (y_test<limi[i+1]))[0]
#     if len(loc) > 0:
#         t = y_test[loc]
#         pml = y_pred[loc]
#         pzr_mayu = zr_mayu[loc]
#         pzr_this = zr_this[loc]
#         print(limi[i],'~',limi[i+1],':',len(loc),file=fff)
#         print('ML:',file=fff)
#         print('  MAE=', np.mean( abs(t-pml) ),file=fff)
#         print('  RMSE=', (np.mean( (t-pml)**2 ))**0.5 ,file=fff)
#         print('Z-R by DSD:',file=fff)
#         print('  MAE=', np.mean( abs(t-pzr_mayu) ),file=fff)
#         print('  RMSE=', (np.mean( (t-pzr_mayu)**2 ))**0.5 ,file=fff)
#         print('Z-R by OB:',file=fff)
#         print('  MAE=', np.mean( abs(t-pzr_this) ),file=fff)
#         print('  RMSE=', (np.mean( (t-pzr_this)**2 ))**0.5 ,file=fff)
#         print('---',file=fff)
# fff.close()
