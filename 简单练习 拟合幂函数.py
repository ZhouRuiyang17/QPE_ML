import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data



#%%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(1, 50),
            nn.Dropout(0.2),
            nn.ReLU(),
            # nn.Linear(50, 50),
            # nn.Dropout(0.2),
            # nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.fc(x)
        return out

net = Net()
#%%

# 缩放
def minmax_scaler(ndarray, _min, _max):
    '''
    ndarray should be 2-D
    _min and _max should be 1-D, not a number
    '''
    _, col = ndarray.shape
    for i in range(col):
        ndarray[:,i] = (ndarray[:,i]-_min[i])/(_max[i]-_min[i])
    
    return ndarray 

x_train = np.arange(0,10,0.1)
y_train = 0.22*x_train**0.53
for i in range(len(x_train)):
    y_train[i] = y_train[i] + np.random.uniform(-0.5,0.5)



x_train = x_train.reshape(100,1)
x_train = minmax_scaler(x_train,[0],[10])

y_train = y_train.reshape(100,1)
y_train = minmax_scaler(y_train,[-2],[2])
#%%

# 张量化+打包+加载
x_train = torch.from_numpy(x_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
train_data = Data.TensorDataset(x_train, y_train)
train_loader = Data.DataLoader(train_data, shuffle=True, num_workers=0)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = nn.MSELoss()
# 训练
epoch_nums = 16
train_loss_list = []
for epoch in range(epoch_nums):
    # 训练过程
    net.train()
    train_loss_list_epoch = []
    
    for step, (x, y_t) in enumerate(train_loader):
        y_p = net(x)
        train_loss = loss_func(y_p, y_t)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_loss_list_epoch.append(train_loss.item())
        
    train_loss_list.append(np.mean(train_loss_list_epoch))
plt.plot(train_loss_list)
plt.show()
#%%
net.eval()
y_pred = net(x_train).detach().numpy()
x_train = x_train.detach().numpy()
y_train = y_train.detach().numpy()

def re_minmax_scaler(ndarray, _min, _max):
    '''
    ndarray should be 2-D
    _min and _max should be 1-D, not a number
    '''
    _, col = ndarray.shape
    for i in range(col):
        # ndarray[:,i] = (ndarray[:,i]-_min[i])/(_max[i]-_min[i])
        ndarray[:,i] = ndarray[:,i] * (_max[i] - _min[i]) + _min[i]
    return ndarray
x_train = re_minmax_scaler(x_train, [0], [10])
y_train = re_minmax_scaler(y_train, [-2], [2])
y_pred = re_minmax_scaler(y_pred, [-2], [2])

plt.scatter(x_train, y_train, label='noise')
plt.scatter(x_train, y_pred,label='pred')
plt.scatter(x_train, 0.22*x_train**0.53, label='truth')
plt.legend()
plt.show()