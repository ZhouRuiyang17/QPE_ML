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
import pandas as pd
from tools import tp_eval, tp_scatter, cm_scatter
import datetime
# ----read
# path = 'history/'+str(datetime.datetime.now())[:10]+'_zs_resample/'
path = 'history/2023-06-18_zs_resample/'

x_train = np.load(path+'x_train.npy')
x_test = np.load(path+'x_test.npy')
x_train = x_train[:,[2,3,4,5]]
x_test = x_test[:,[2,3,4,5]]



y_train = np.load(path+'y_train.npy').reshape(-1,1)
y_test = np.load(path+'y_test.npy')


# ----标准化：x_test
# pp_x = MinMaxScaler()
# _ = pp_x.fit(x_train)
# x_test = pp_x.transform(x_test)

# pp_y = MinMaxScaler()
# _ = pp_y.fit(y_train)

#%%
def min_max(data, mini=None, maxi=None):
    data[data<mini] = mini
    data[data>maxi] = maxi
    
    row, col = data.shape
    newdata = np.zeros(shape=(row, col))
    for i in range(col):
        # newdata[:,i] = (data[:,i]-np.min(data[:,i]))/(np.max(data[:,i])-np.min(data[:,i]))
        newdata[:,i] = (data[:,i]-mini)/(maxi-mini)
    return newdata
x_train = min_max(x_train.copy(), 0, 75)
x_test = min_max(x_test.copy(), 0, 75)





# ----张量化：x_test
x_test = torch.from_numpy(x_test.astype(np.float32))


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

# ----载入模型
net = torch.load(path+'net_best.pt')
net.eval()

# ----估计y_pred并逆张量化y_pred和x_test
y_pred = net(x_test).detach().numpy()
x_test = x_test.detach().numpy()
# ----逆标准化：y_pred和x_test
def min_max_rev(data, mini, maxi):
    row, col = data.shape
    newdata = np.zeros(shape=(row, col))
    for i in range(col):
        # newdata[:,i] = (data[:,i]-np.min(data[:,i]))/(np.max(data[:,i])-np.min(data[:,i]))
        # newdata[:,i] = (data[:,i]-mini)/(maxi-mini)
        newdata[:,i] = data[:,i] * (maxi-mini) + mini
    return newdata


y_pred = min_max_rev(y_pred, 0, 100).flatten()
x_test = min_max_rev(x_test, 0, 75)

# y_pred = pp_y.inverse_transform(y_pred)
# y_pred = y_pred.flatten()
# x_test = pp_x.inverse_transform(x_test)

#%%
# ----其他模型
z = x_test[:,1]
zr_mayu = 0.0576*(10**(z/10))**0.557
b = 1/1.4; a = (1/300)**b
zr_300  = a     *(10**(z/10))**b
b = 1/1.6; a = (1/200)**b
zr_200  = a     *(10**(z/10))**b




#%% 统计分析


# ----所有
df = pd.DataFrame(np.zeros(shape=(5,5)))
df.iloc[:, 0] = str(len(y_test)), 'BIAS', 'ME', 'RMSE', 'CORR'
df.iloc[0,1:] = 'MLP', 'Z=238R^1.57', 'Z=300R^1.4', 'Z=200R^1.6' 
df.iloc[1:, 1] = tp_eval(y_test, y_pred)
df.iloc[1:, 2] = tp_eval(y_test, zr_mayu)
df.iloc[1:, 3] = tp_eval(y_test, zr_300)
df.iloc[1:, 4] = tp_eval(y_test, zr_200)
df.to_csv(path+'all.csv')

# ----分段
limi = [0, 0.1, 10, 25, 50, 1000]
for i in range(5):
    loc = np.where((y_test>=limi[i]) & (y_test<limi[i+1]))[0]
    # print(len(loc))
    
    df1 = pd.DataFrame(np.zeros(shape=(5,5)))
    df1.iloc[:, 0] = str(len(loc)), 'BIAS', 'ME', 'RMSE', 'CORR'
    df1.iloc[0,1:] = 'MLP', 'Z=238R^1.57', 'Z=300R^1.4', 'Z=200R^1.6'  
    df1.iloc[1:, 1] = tp_eval(y_test[loc], y_pred[loc])
    df1.iloc[1:, 2] = tp_eval(y_test[loc], zr_mayu[loc])
    df1.iloc[1:, 3] = tp_eval(y_test[loc], zr_300[loc])
    df1.iloc[1:, 4] = tp_eval(y_test[loc], zr_200[loc])
    df1.to_csv(path+str(limi[i])+'-'+str(limi[i+1])+'.csv')

# ----画图


stat=[df.iloc[0,0],df.iloc[1,1],df.iloc[2,1],df.iloc[3,1],df.iloc[4,1]]
tp_scatter(y_test, y_pred, label='$MLP$',
            xylabel = 'Rain rate (mm/h)', minmax=[0,100],
            num='(a)',stat=stat)
stat=[df.iloc[0,0],df.iloc[1,2],df.iloc[2,2],df.iloc[3,2],df.iloc[4,2]]
tp_scatter(y_test, zr_mayu, label='$Z=238R^{1.57}$',
            xylabel = 'Rain rate (mm/h)', minmax=[0,100],
            num='(b)',stat=stat)
stat=[df.iloc[0,0],df.iloc[1,3],df.iloc[2,3],df.iloc[3,3],df.iloc[4,3]]
tp_scatter(y_test, zr_300, label='$Z=300R^{1.4}$',
            xylabel = 'Rain rate (mm/h)', minmax=[0,100],
            num='(c)',stat=stat)
stat=[df.iloc[0,0],df.iloc[1,4],df.iloc[2,4],df.iloc[3,4],df.iloc[4,4]]
tp_scatter(y_test, zr_200, label='$Z=200R^{1.6}$',
            xylabel = 'Rain rate (mm/h)', minmax=[0,100],
            num='(d)',stat=stat)



# cm_scatter(x_test[:,1], y_test, label='obs',
#             xlabel='Z (dB)', ylabel='R (mm/h)', xminmax=[0,80], yminmax=[0,100],
#             num='(a)')
# cm_scatter(x_test[:,1], y_pred, label='$MLP$',
#             xlabel='Z (dB)', ylabel='R (mm/h)', xminmax=[0,80], yminmax=[0,100],
#             num='(b)')
# cm_scatter(x_test[:,1], zr_mayu, label='$Z=238R^{1.57}$',
#             xlabel='Z (dB)', ylabel='R (mm/h)', xminmax=[0,80], yminmax=[0,100],
#             num='(c)')
# cm_scatter(x_test[:,1], zr_300, label='$Z=300R^{1.4}$',
#             xlabel='Z (dB)', ylabel='R (mm/h)', xminmax=[0,80], yminmax=[0,100],
#             num='(d)')
# cm_scatter(x_test[:,1], zr_200, label='$Z=200R^{1.6}$',
#             xlabel='Z (dB)', ylabel='R (mm/h)', xminmax=[0,80], yminmax=[0,100],
#             num='(e)')

# plt.boxplot([(y_pred - y_test),(zr_mayu - y_test),(zr_300 - y_test),(zr_200 - y_test)], 
#             labels = ['$MLP$', '$Z=238R^{1.57}$', '$Z=300R^{1.4}$', '$Z=200R^{1.6}$'],
#             showfliers=True, showmeans = True, meanline = True)
# plt.grid()
# plt.show()

plt.boxplot([(y_pred - y_test),(zr_mayu - y_test),(zr_300 - y_test),(zr_200 - y_test)], 
            labels = ['$MLP$', '$Z=238R^{1.57}$', '$Z=300R^{1.4}$', '$Z=200R^{1.6}$'],
            showfliers=False, showmeans = True, meanline = True)
plt.grid()
plt.title('Estimate - observation')
plt.ylabel('mm/h')
plt.show()

#%%
# plt.boxplot([(y_pred - y_test),(zr_300 - y_test)], 
#             labels = ['$MLP$', '$Z=300R^{1.4}$'],
#             showfliers=False, showmeans = True, meanline = True, widths = 0.715)
# plt.title('Estimate - observation')
# plt.ylabel('mm/h')
# plt.grid()
# plt.show()
#%%

# x_test = np.load(path+'x_test.npy')
# df = pd.DataFrame(x_test)
# df = pd.concat([df,pd.DataFrame(y_test)], axis=1)
# df = pd.concat([df,pd.DataFrame(y_pred)], axis=1)
# df = pd.concat([df,pd.DataFrame(zr_mayu)], axis=1)
# df = pd.concat([df,pd.DataFrame(zr_300)], axis=1)
# df = pd.concat([df,pd.DataFrame(zr_200)], axis=1)
# df.to_excel(path+'test.xlsx')
#%%


