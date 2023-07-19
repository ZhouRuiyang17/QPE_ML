import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler, QuantileTransformer
from scipy.stats import kstest
from scipy import stats
# *****************************************************[1] 准备数据*****************************************************
# ----read dataset
import datetime
# path = 'history/'+str(datetime.datetime.now())[:10]+'_zs_resample/'
path = r'E:\QPE_ML\history\2023-05-27_zs_resample/'
x_train = np.load(path+'x_train.npy')
x_vali = np.load(path+'x_vali.npy')
x_test = np.load(path+'x_test.npy')

# ----only use Z at 0.5, 1.5, 2.5, 3.5 km
x_train = x_train[:,[2,3,4,5]]
x_vali = x_vali[:,[2,3,4,5]]
x_test = x_test[:,[2,3,4,5]]


y_train = np.load(path+'y_train.npy').reshape(-1,1)
y_vali = np.load(path+'y_vali.npy').reshape(-1,1)
y_test = np.load(path+'y_test.npy').reshape(-1,1)




#%%
# ----scaler

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
x_vali = min_max(x_vali.copy(), 0, 75)
x_test = min_max(x_test.copy(), 0, 75)


y_train = min_max(y_train.copy(), 0, 100)
y_vali = min_max(y_vali.copy(), 0, 100)
y_test = min_max(y_test.copy(), 0, 100)

# pp_x = MinMaxScaler()
# _ = pp_x.fit(x_train)
# x_train = pp_x.transform(x_train)
# x_vali = pp_x.transform(x_vali)
# x_test = pp_x.transform(x_test)

# pp_y = MinMaxScaler()
# _ = pp_y.fit(y_train)
# y_train = pp_y.transform(y_train)
# y_vali = pp_y.transform(y_vali)
# y_test = pp_y.transform(y_test)

#%%

x_train = torch.from_numpy(x_train.astype(np.float32))
x_vali = torch.from_numpy(x_vali.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_vali = torch.from_numpy(y_vali.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))


train_data = Data.TensorDataset(x_train, y_train)
train_loader = Data.DataLoader(train_data, shuffle=True, batch_size=16, num_workers=0)

vali_data = Data.TensorDataset(x_vali, y_vali)
vali_loader = Data.DataLoader(vali_data, shuffle=True, batch_size=16, num_workers=0)

test_data = Data.TensorDataset(x_test, y_test)
test_loader = Data.DataLoader(test_data, shuffle=True, batch_size=16, num_workers=0)



#%%
# *****************************************************[2] 定义网络和训练*****************************************************

# ----4-layer MLP
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc = nn.Sequential(
            
            # 6: 4*z + lon + lat
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

# net = Net()
net = torch.load(path+'net_best.pt')

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = nn.MSELoss()



epoch_nums = 2**6
train_loss_list, vali_loss_list = [], []
with open(path+'loss.txt', 'w') as f:
    for epoch in range(epoch_nums):
        # training
        net.train()
        train_loss_all = 0.0
        train_num = 0

        for step, (x, y_t) in enumerate(train_loader):
            b, _ = x.shape
            if b != 16:
                break
            y_p = net(x)
            train_loss = loss_func(y_p, y_t)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_loss_all += train_loss.item()
            train_num += 1

        train_loss_list.append(train_loss_all / train_num)

        # validating
        net.eval()
        vali_loss_all = 0.0
        vali_num = 0

        for step, (x, y_t) in enumerate(vali_loader):
            b, _ = x.shape
            if b != 16:
                break
            y_p = net(x)
            vali_loss = loss_func(y_p, y_t)

            vali_loss_all += vali_loss.item()
            vali_num += 1

        vali_loss_list.append(vali_loss_all / vali_num)

        if len(vali_loss_list) == 1:
            loss_min = vali_loss_list[-1]
            torch.save(net, path+'net_best.pt')
            print(epoch)
        elif len(vali_loss_list) > 1:
            if loss_min >= vali_loss_list[-1]:
                loss_min = vali_loss_list[-1]
                torch.save(net, path+'net_best.pt')
                print(epoch)
                

        f.write('epoch:' + str(epoch) + ', train_loss:' + str(train_loss_list[-1]) + ', vali_loss:' + str(vali_loss_list[-1]) + '\n')
f.close()


plt.plot(range(epoch_nums), train_loss_list, label='train loss')
plt.plot(range(epoch_nums), vali_loss_list, label='vali loss')
plt.title('Train And Vali Loss')
plt.legend()
plt.show()
