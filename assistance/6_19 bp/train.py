import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



#读取数据集
df = pd.read_csv('QPE.csv')
# df = df[['PR_CUM'] < 100]
data = np.array(df)
data_x, data_y = data[:, 1:-1], data[:, -1]

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.1, random_state=123)

scales = MinMaxScaler(feature_range=(0, 1))
x_train = scales.fit_transform(x_train)
x_test = scales.transform(x_test)
y_train = scales.fit_transform(y_train.reshape(-1, 1))
y_test = scales.transform(y_test.reshape(-1, 1))




# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(4, 50),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.fc(x)
        return out

net = Net()
# 将数据转化为张量
x_train_nots = torch.from_numpy(x_train.astype(np.float32))
x_test_nots = torch.from_numpy(x_test.astype(np.float32))
y_train_t = torch.from_numpy(y_train.reshape(-1, 1).astype(np.float32))
y_test_t = torch.from_numpy(y_test.reshape(-1, 1).astype(np.float32))

# 定义数据加载器
train_data_nots = Data.TensorDataset(x_train_nots, y_train_t)
train_nots_loader = Data.DataLoader(train_data_nots, shuffle=True, batch_size=64, num_workers=0)

test_data_nots = Data.TensorDataset(x_test_nots, y_test_t)
test_nots_loader = Data.DataLoader(test_data_nots, shuffle=True, batch_size=64, num_workers=0)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = nn.MSELoss()

# 定义epcoh_nums
epoch_nums = 15

train_loss_list, test_loss_list = [], []
train_acc_list, test_acc_list = [], []
with open('loss.txt', 'w') as f:
    for epoch in range(epoch_nums):
        # 训练过程
        net.train()
        train_loss_all = 0.0
        train_num = 0

        for step, (x, y_t) in enumerate(train_nots_loader):
            b, _ = x.shape
            if b != 64:
                break
            y_p = net(x)

            train_loss = loss_func(y_p, y_t)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_loss_all += train_loss.item()
            train_num += 1

        train_loss_list.append(train_loss_all / train_num)
        print('train epoch:{} loss:{:.6f}'.format(epoch, train_loss_all / train_num))

        # 测试过程
        net.eval()
        test_loss_all = 0.0
        test_num = 0

        for step, (x, y_t) in enumerate(test_nots_loader):
            b, _ = x.shape
            if b != 64:
                break
            y_p = net(x)
            test_loss = loss_func(y_p, y_t)

            test_loss_all += test_loss.item()
            test_num += 1

        test_loss_list.append(test_loss_all / test_num)
        print('test epoch:{} loss:{:.6f}'.format(epoch, test_loss_all / test_num))
        f.write('epoch:' + str(epoch) + ', train_loss:' + str(train_loss_list[-1]) + ', test_loss:' + str(test_loss_list[-1]) + '\n')
f.close()


net.eval()
# data_x = scales.fit_transform(data_x)
# data_y = scales.fit_transform(data_y.reshape(-1, 1))
# data_x = torch.from_numpy(data_x.astype(np.float32))
y = list(net(x_test_nots).detach().numpy().reshape(-1))
data_y = list(y_test_t.reshape(-1))
print(len(y))

# 可视化
plt.figure(figsize=(16, 7))
plt.plot(range(1000), data_y[:1000], label='true')
plt.plot(range(1000), y[:1000], label='pre')
plt.title('True And Pre Value')
plt.legend()
plt.show()

plt.figure(figsize=(16, 7))
plt.plot(range(epoch_nums), train_loss_list, label='train loss')
plt.plot(range(epoch_nums), test_loss_list, label='test loss')
plt.title('Train And Test Loss')
plt.legend()
plt.show()

