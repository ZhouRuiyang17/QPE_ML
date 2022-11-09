# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 15:07:17 2022

@author: admin
"""

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

fname = 'composite/035330.nc'
f = nc.Dataset(fname, 'r')
print(f)
print(f.variables['DBZ'])
zh = np.array(f.variables['DBZ'])
zh[0, 0, 400:500, 300:600] = np.nan
plt.pcolor(np.arange(800), np.arange(800)-400, zh[0, 0])
plt.show()
# plt.pcolor(np.arange(300), np.arange(100)-400, zh[0, 0, 400:500, 300:600])
# plt.show()
# plt.pcolor(zh[0, 0, 400:500, 300:600])
# plt.pcolor(zh[0, 2, 400:500, 300:600])

# vpr = zh[1, :, x, y]