import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import os

# ----定义结构
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

# ----标准化和逆标准化
def min_max(data, mini, maxi):
    data[data<mini] = mini
    data[data>maxi] = maxi
    
    row, col = data.shape
    newdata = np.zeros(shape=(row, col))
    for i in range(col):
        newdata[:,i] = (data[:,i]-mini)/(maxi-mini)
    return newdata
def min_max_rev(data, mini, maxi):
    row, col = data.shape
    newdata = np.zeros(shape=(row, col))
    for i in range(col):
        newdata[:,i] = data[:,i] * (maxi-mini) + mini
    return newdata

def mlp(data, net):
    # ----张量化
    tensor = torch.from_numpy(data.astype(np.float32))
    # ----载入模型
    net.eval()
    # ----计算
    pred = net(tensor).detach().numpy()
    pred = min_max_rev(pred, 0, 100).flatten()
    
    return pred


#%%
if __name__ == "__main__":
    # ----加载模型
    path = 'E:/QPE_ML/history/2023-05-27_zs_resample/'# !!!
    net = torch.load(path+'net_best.pt')
    method = input("method?")
    
    # ----日期
    # ls_d = os.listdir('D:/data/银川/rpg/')
    ls_d = os.listdir('H:/SA雷达三维拼图和组合反射率因子拼图数据/Z三维拼图20层800X800格点/')
    for date in ls_d[:]:
        if date != '20180807':
            # print('skip '+date)
            continue
        print(date)
        # ----日期下的文件
        # path = 'D:/data/银川/mosaic/'+date+'/'; ls_f = os.listdir(path)
        path = 'H:/SA雷达三维拼图和组合反射率因子拼图数据/Z三维拼图20层800X800格点/'+date+'/'; ls_f = os.listdir(path)
        for fname in ls_f[:]:
            # fpath = path + fname
            # grid = np.load(fpath); qpe = np.zeros(shape=(400,400))
            if fname.endswith('nc'):
                fpath = path + fname
                import netCDF4 as nc
                grid = nc.Dataset(fpath,'r'); grid = np.array(grid.variables['DBZ'])[0,:4,:,:]
                qpe = np.zeros(shape=(800,800))
            else:
                continue
        # grid = np.load('D:/data/银川/mosaic/20200830/Z_RADR_I_Z9951_20200830034000_O_DOR_CA_CAP.npy')
        # plt.contourf(grid[1])
        # plt.colorbar()
        # plt.show()
            
            # ----计算
            if method == 'zr':
                b = 1/1.4; a = (1/300)**b
                qpe_zr = a*( 10**(grid[1]/10) )**b
                qpe = qpe_zr
            # plt.contourf(qpe_zr)
            # plt.colorbar()
            # plt.show()
            elif method == 'mlp':
                qpe_mlp = np.zeros_like(qpe)
                counter_x = []; counter_y = []; inputs = []
                # for ix in range(400):
                #     for iy in range(400):
                for ix in np.arange(270, 530):
                    for iy in np.arange(310, 650):
                        input1 = grid[:, iy, ix]
                        
                        num_ref = 0
                        for i in range(4):
                            if input1[i] > 0:
                                num_ref += 1
                                
                        # if input1[0] > 0 or input1[1] > 0 or input1[2] > 0 or input1[3] > 0:
                        if num_ref >= 2:
                            input1 = input1.reshape(1, -1)
                            input2 = min_max(input1, 0, 75)
                            counter_x.append(ix); counter_y.append(iy)
                            inputs.append(input2)
                inputs = np.array(inputs); inputs = np.squeeze(inputs)
                rr = mlp(inputs, net)
                qpe_mlp[counter_y, counter_x] = rr
                qpe = qpe_mlp
                # count = 0
                # for ix in range(400):
                #     for iy in range(400):
                #         qpe_mlp[iy, ix] = rr[count]
                #         count += 1
            # plt.contourf(qpe_mlp)
            # plt.colorbar()
            # plt.show()
            
            # qpe[0] = qpe_zr; qpe[1] = qpe_mlp
            # ----存储
            # save_path = path.replace('mosaic', 'rain_rate_'+method)
            save_path = path + 'rain_rate_'+method + '/'
            if os.path.exists(save_path) == False:
                os.makedirs(save_path)
            new_fpath = save_path + fname
            np.save(new_fpath, qpe)
    