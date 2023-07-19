import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random

# 【0】----raw data saved as .xlsx file
df = pd.read_excel('dataset_ref55.xlsx', index_col = 0)
ref55 = df['refs'].iloc[1]
# data = df.iloc[:,1:].values.astype(np.float64)


#%%
# ----for faster reading, save raw data as .npy file
data = np.load('data_zs_221030_utc8.npy')

# ----select the rainy points
loc = np.where(data[:,-1] > 0)
data_rain = data[loc]
loc = np.where( (data_rain[:,2] > 0) | (data_rain[:,3] > 0) | (data_rain[:,4] > 0) | (data_rain[:,5] > 0) )
data_rain = data_rain[loc]


# ----classify the rain rate
data_rain_1 = data_rain[data_rain[:,-1]<10] # small
data_rain_2 = data_rain[(data_rain[:,-1]>=10) & (data_rain[:,-1]<25)]# middle
data_rain_3 = data_rain[(data_rain[:,-1]>=25) & (data_rain[:,-1]<50)]# heavy
data_rain_4 = data_rain[(data_rain[:,-1]>=50)]# severe

# ----resample the small rain to improve the distribution of rain rate
data_rain_1 = np.array(random.sample(data_rain_1.tolist(), len(data_rain_2)*2))


# ----divide dataset for training, validation and test: 6:1:3
def spliter(x,y,test_size):
    x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=test_size,random_state=123)
    return  x_train, x_test, y_train, y_test




x,x1_test,y,y1_test=spliter(data_rain_1[:,:-1],data_rain_1[:,-1],0.3)
x1_train,x1_vali,y1_train,y1_vali=spliter(x,y,1/7)


x,x2_test,y,y2_test=spliter(data_rain_2[:,:-1],data_rain_2[:,-1],0.3)
x2_train,x2_vali,y2_train,y2_vali=spliter(x,y,1/7)


x,x3_test,y,y3_test=spliter(data_rain_3[:,:-1],data_rain_3[:,-1],0.3)
x3_train,x3_vali,y3_train,y3_vali=spliter(x,y,1/7)


x,x4_test,y,y4_test=spliter(data_rain_4[:,:-1],data_rain_4[:,-1],0.3)
x4_train,x4_vali,y4_train,y4_vali=spliter(x,y,1/7)



x_train=np.vstack((x1_train,x2_train,x3_train,x4_train))
x_vali=np.vstack((x1_vali,x2_vali,x3_vali,x4_vali))
x_test=np.vstack((x1_test,x2_test,x3_test,x4_test))



y_train=np.hstack((y1_train,y2_train,y3_train,y4_train))
y_vali=np.hstack((y1_vali,y2_vali,y3_vali,y4_vali))
y_test=np.hstack((y1_test,y2_test,y3_test,y4_test))

# ---- shuffle
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


# ----plot the histgram and the scatter to check distribution

# hist or rain rate
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

# Z at 1.5km vs R
plt.scatter(x_train[:,3], y_train)
plt.xlabel('Z (dBZ)')
plt.ylabel('R (mm)')
plt.title('train')
plt.show()

plt.scatter(x_vali[:,3], y_vali)
plt.xlabel('Z (dBZ)')
plt.ylabel('R (mm)')
plt.title('vali')
plt.show()

plt.scatter(x_test[:,3], y_test)
plt.xlabel('Z (dBZ)')
plt.ylabel('R (mm)')
plt.title('test')
plt.show()
#%% 保存数据
import os
import datetime;path = 'ref55/'+str(datetime.datetime.now())[:10]+'/'
if os.path.exists(path) == False:
    os.mkdir(path)
    print(1)
np.save(path+'/x_train.npy',x_train)
np.save(path+'/x_vali.npy',x_vali)
np.save(path+'/x_test.npy',x_test)
np.save(path+'/y_train.npy',y_train)
np.save(path+'/y_vali.npy',y_vali)
np.save(path+'/y_test.npy',y_test)