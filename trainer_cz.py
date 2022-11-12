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
# ----读取数据集
path = 'result/'
x_train = np.load(path+'x_train.npy')[:,-1:]#.reshape(-1,1)
x_vali = np.load(path+'x_vali.npy')[:,-1:]#.reshape(-1,1)
x_test = np.load(path+'x_test.npy')[:,-1:]#.reshape(-1,1)
y_train = np.load(path+'y_train.npy').reshape(-1,1)
y_vali = np.load(path+'y_vali.npy').reshape(-1,1)
y_test = np.load(path+'y_test.npy').reshape(-1,1)


#%%
# ----映射

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

#%%
# ----张量化
x_train = torch.from_numpy(x_train.astype(np.float32))
x_vali = torch.from_numpy(x_vali.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_vali = torch.from_numpy(y_vali.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# ----打包+加载
train_data = Data.TensorDataset(x_train, y_train)
train_loader = Data.DataLoader(train_data, shuffle=True, batch_size=16, num_workers=0)

vali_data = Data.TensorDataset(x_vali, y_vali)
vali_loader = Data.DataLoader(vali_data, shuffle=True, batch_size=16, num_workers=0)

test_data = Data.TensorDataset(x_test, y_test)
test_loader = Data.DataLoader(test_data, shuffle=True, batch_size=16, num_workers=0)



#%%
# *****************************************************[2] 定义网络和训练*****************************************************

# ----定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc = nn.Sequential(
            
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()

        )

    def forward(self, x):
        out = self.fc(x)
        return out

net = Net()


# ----定义优化器和损失函数
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = nn.MSELoss()



# 训练
epoch_nums = 2**5
train_loss_list, vali_loss_list = [], []
with open(path+'loss.txt', 'w') as f:
    for epoch in range(epoch_nums):
        # 训练过程
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

        # 验证过程
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
