import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler



df = pd.read_excel('e:/20180716dsd.xlsx')
data_rain = df.iloc[:,1:].values.astype(np.float64)

# np.save('data_zs_221030_utc8.npy', data)
# loc = np.where((data[:,1] > 201907280000) & (data[:,1] < 201908010000))
# d = data[loc]
# np.save('data_zs_190728_190731.npy', d)
# data = data[:314720] # 2023.2.16 缩短TVT集，留一场用于实践
# np.save('data_zs_221030_utc8_clip.npy', data)


# data = np.load('data_zs_221030_utc8.npy')

# #%% 数据清洗

# # ----下雨
# loc = np.where(data[:,2] > 0)
# data_rain = data[loc]

# # ----清洗
# pred = 0.0576*(10**(data_rain[:,3]/10))**0.557# site，ts，0.5，1.5：0，1，2，3
# true = data_rain[:,-1]
# loc = np.where((abs(pred-true)/true)<1)
# data_rain_slct = data_rain[loc]

#%% 划分降雨等级
# loc=np.where((data[:,-1]>0))
# data_rain=data[loc]
# loc=np.where(data[:,-1]==0)
# data_norain=data[loc]

# # ----减少不下雨的数量
# ls=np.arange(len(data_norain))
# loc=np.random.choice(ls,int(len(data_rain)/10))# 从ls中随机取 int(len(data_rain)/10) 个
# data_norain2=data_norain[loc]

# ----划分降雨等级
data_rain_1 = data_rain[data_rain[:,2]<10]
data_rain_2 = data_rain[(data_rain[:,2]>=10) & (data_rain[:,2]<25)]
data_rain_3 = data_rain[(data_rain[:,2]>=25) & (data_rain[:,2]<50)]
data_rain_4 = data_rain[(data_rain[:,2]>=50)]
#%% 划分数据集
def spliter(x,y,test_size):
    x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=test_size,random_state=123)
    return  x_train, x_test, y_train, y_test

# # ----不下雨
# x,x0_test,y,y0_test=spliter(data_norain2[:,:-1],data_norain2[:,-1],0.3)
# x0_train,x0_vali,y0_train,y0_vali=spliter(x,y,1/7)

# ----小雨
x,x1_test,y,y1_test=spliter(data_rain_1[:,3:6],data_rain_1[:,2],0.3)
x1_train,x1_vali,y1_train,y1_vali=spliter(x,y,1/7)

# ----中雨
x,x2_test,y,y2_test=spliter(data_rain_2[:,3:6],data_rain_2[:,2],0.3)
x2_train,x2_vali,y2_train,y2_vali=spliter(x,y,1/7)

# ----大雨
x,x3_test,y,y3_test=spliter(data_rain_3[:,3:6],data_rain_3[:,2],0.3)
x3_train,x3_vali,y3_train,y3_vali=spliter(x,y,1/7)

# ----暴雨
x,x4_test,y,y4_test=spliter(data_rain_4[:,3:6],data_rain_4[:,2],0.3)
x4_train,x4_vali,y4_train,y4_vali=spliter(x,y,1/7)

# ---合并
# x_train=np.vstack((x0_train,x1_train,x2_train,x3_train,x4_train))
# x_vali=np.vstack((x0_vali,x1_vali,x2_vali,x3_vali,x4_vali))
# x_test=np.vstack((x0_test,x1_test,x2_test,x3_test,x4_test))

x_train=np.vstack((x1_train,x2_train,x3_train,x4_train))
x_vali=np.vstack((x1_vali,x2_vali,x3_vali,x4_vali))
x_test=np.vstack((x1_test,x2_test,x3_test,x4_test))

# y_train=np.hstack((y0_train,y1_train,y2_train,y3_train,y4_train))
# y_vali=np.hstack((y0_vali,y1_vali,y2_vali,y3_vali,y4_vali))
# y_test=np.hstack((y0_test,y1_test,y2_test,y3_test,y4_test))

y_train=np.hstack((y1_train,y2_train,y3_train,y4_train))
y_vali=np.hstack((y1_vali,y2_vali,y3_vali,y4_vali))
y_test=np.hstack((y1_test,y2_test,y3_test,y4_test))
# ---打乱
ls=np.arange(len(x_train))
np.random.shuffle(ls)
x_train=x_train[ls]
y_train=y_train[ls]

ls=np.arange(len(x_vali))
np.random.shuffle(ls)
x_vali=x_vali[ls]
y_vali=y_vali[ls]

ls=np.arange(len(x_test))
np.random.shuffle(ls)
x_test=x_test[ls]
y_test=y_test[ls]


#%% 画图

# ----降雨的直方图
plt.hist(y_train,bins=np.arange(0,200,10))
plt.yscale('log')
plt.title('train')
plt.show()

plt.hist(y_vali,bins=np.arange(0,200,10))
plt.yscale('log')
plt.title('vali')
plt.show()

plt.hist(y_test,bins=np.arange(0,200,10))
plt.yscale('log')
plt.title('test')
plt.show()

# # ----Z-R散点图
# plt.scatter(x_train[:,2], y_train)
# # plt.ylim(0,100)
# # plt.xlim(-30,70)
# plt.xlabel('Z (dBZ)')
# plt.ylabel('R (mm)')
# plt.title('train')
# plt.show()

# plt.scatter(x_vali[:,2], y_vali)
# # plt.ylim(0,100)
# # plt.xlim(-30,70)
# plt.xlabel('Z (dBZ)')
# plt.ylabel('R (mm)')
# plt.title('vali')
# plt.show()

# plt.scatter(x_test[:,2], y_test)
# # plt.ylim(0,100)
# # plt.xlim(-30,70)
# plt.xlabel('Z (dBZ)')
# plt.ylabel('R (mm)')
# plt.title('test')
# plt.show()
#%% 保存数据
import os
dire = 'result_dp_dsd'
if dire not in os.listdir('./'):
    os.mkdir(dire)
    print(1)
np.save(dire+'/x_train.npy',x_train)
np.save(dire+'/x_vali.npy',x_vali)
np.save(dire+'/x_test.npy',x_test)
np.save(dire+'/y_train.npy',y_train)
np.save(dire+'/y_vali.npy',y_vali)
np.save(dire+'/y_test.npy',y_test)