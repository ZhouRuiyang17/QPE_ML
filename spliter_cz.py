'''
适用于：label+x+y的格式
2022.7.22
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
df = pd.read_csv('dataset_cz.csv')
#%%
data = df.iloc[:,-4:].values.astype(np.float64)

# 预处理
loc=np.where(data[:,-1]>0)
data_rain=data[loc]
loc=np.where(data[:,-1]==0)
data_norain=data[loc]

# ----减少不下雨的数量
ls=np.arange(len(data_rain))
loc=np.random.choice(ls,int(len(data_rain)/10))
data_norain2=data_norain[loc]

# ----划分降雨等级
data_rain_1=data_rain[data_rain[:,-1]<10]
data_rain_2=data_rain[(data_rain[:,-1]>=10) & (data_rain[:,-1]<25)]
data_rain_3=data_rain[(data_rain[:,-1]>=25) & (data_rain[:,-1]<50)]
data_rain_4=data_rain[(data_rain[:,-1]>=50)]
#%%
# 划分数据集
def spliter(x,y,test_size):
    x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=test_size,random_state=123)
    return  x_train, x_test, y_train, y_test

# ----不下雨
x,x0_test,y,y0_test=spliter(data_norain2[:,:-1],data_norain2[:,-1],0.1)
x0_train,x0_vali,y0_train,y0_vali=spliter(x,y,1/9)

# ----小雨
x,x1_test,y,y1_test=spliter(data_rain_1[:,:-1],data_rain_1[:,-1],0.1)
x1_train,x1_vali,y1_train,y1_vali=spliter(x,y,1/9)

# ----中雨
x,x2_test,y,y2_test=spliter(data_rain_2[:,:-1],data_rain_2[:,-1],0.1)
x2_train,x2_vali,y2_train,y2_vali=spliter(x,y,1/9)

# ----大雨
x,x3_test,y,y3_test=spliter(data_rain_3[:,:-1],data_rain_3[:,-1],0.1)
x3_train,x3_vali,y3_train,y3_vali=spliter(x,y,1/9)

# ----暴雨
x,x4_test,y,y4_test=spliter(data_rain_4[:,:-1],data_rain_4[:,-1],0.1)
x4_train,x4_vali,y4_train,y4_vali=spliter(x,y,1/9)

# ---合并
x_train=np.vstack((x0_train,x1_train,x2_train,x3_train,x4_train))
x_vali=np.vstack((x0_vali,x1_vali,x2_vali,x3_vali,x4_vali))
x_test=np.vstack((x0_test,x1_test,x2_test,x3_test,x4_test))

y_train=np.hstack((y0_train,y1_train,y2_train,y3_train,y4_train))
y_vali=np.hstack((y0_vali,y1_vali,y2_vali,y3_vali,y4_vali))
y_test=np.hstack((y0_test,y1_test,y2_test,y3_test,y4_test))
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



import matplotlib.pyplot as plt
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


plt.scatter(x_train[:,-1], y_train)
plt.ylim(0,100)
plt.xlim(-30,70)
plt.xlabel('Z (dBZ)')
plt.ylabel('R (mm)')
plt.title('train')
plt.show()

plt.scatter(x_vali[:,-1], y_vali)
plt.ylim(0,100)
plt.xlim(-30,70)
plt.xlabel('Z (dBZ)')
plt.ylabel('R (mm)')
plt.title('vali')
plt.show()

plt.scatter(x_test[:,-1], y_test)
plt.ylim(0,100)
plt.xlim(-30,70)
plt.xlabel('Z (dBZ)')
plt.ylabel('R (mm)')
plt.title('test')
plt.show()


#%%
# 保存数据
import os
dire = 'result'
if dire not in os.listdir('./'):
    os.mkdir('result')
    print(1)
np.save(dire+'/x_train.npy',x_train)
np.save(dire+'/x_vali.npy',x_vali)
np.save(dire+'/x_test.npy',x_test)
np.save(dire+'/y_train.npy',y_train)
np.save(dire+'/y_vali.npy',y_vali)
np.save(dire+'/y_test.npy',y_test)
