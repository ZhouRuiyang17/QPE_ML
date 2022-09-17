'''
适用于：label+x+y的格式
层状雨
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
df = pd.read_csv('dataset_cz.csv')
#%%
data = df.iloc[:,-4:].values.astype(np.float64)

loc=np.where((data[:,-2]<=35) & (data[:,-2]>0) & (data[:,-1]>=0))
data=data[loc]

#%%
# 划分数据集
def spliter(x,y,test_size):
    x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=test_size,random_state=123)
    return  x_train, x_test, y_train, y_test


x,x_test,y,y_test=spliter(data[:,:-1],data[:,-1],0.1)
x_train,x_vali,y_train,y_vali=spliter(x,y,1/9)





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
