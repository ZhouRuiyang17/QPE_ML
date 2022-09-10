# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 21:38:30 2022

@author: admin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_excel('data_zs_utc8_ave6.xlsx').iloc[:,1:].values
#%%
data[data==-999]=np.nan
stnms = data[:20,0]
for stnm in stnms[:]:
    loc = np.where(data[:,0] == stnm)
    data_stnm = data[loc]
    plt.scatter(data_stnm[:,2], data_stnm[:,8])
    plt.title(stnm)
    plt.show()
#%%
site = 54511
loc = np.where(data[:,0] == site)
data_select = data[loc]

site = 54594
loc = np.where(data[:,0] == site)
data_select = np.vstack([data_select, data[loc]])

# site = 54431
# loc = np.where(data[:,0] == site)
# data_select = np.vstack([data_select, data[loc]])

plt.scatter(data_select[:,2], data_select[:,8])
plt.show()