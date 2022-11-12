import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%% 简单拼接

# labels = pd.read_excel('labels_cz.xlsx',index_col=0)
# radar = pd.read_excel('cz0-86.xlsx',index_col=0)
# rr = pd.read_excel('rr_cz.xlsx',index_col=0)
# data = pd.concat([labels, radar, rr], axis = 1)
# data.to_excel('data_cz_utc8_ave6.xlsx')

#%% 地空对应检查
data = pd.read_excel('data_cz_utc8_ave6.xlsx',incdex_col = 0).values

data[data==-999]=np.nan
stnms = data[:20,0]
for stnm in stnms[:]:
    loc = np.where(data[:,0] == stnm)
    data_stnm = data[loc]
    plt.scatter(data_stnm[:,2], data_stnm[:,-1])
    plt.title(stnm)
    plt.show()
