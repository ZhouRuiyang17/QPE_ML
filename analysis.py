import mytools
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
#%%
if __name__ == "__main__":
    
    # ----工作目录
    path = 'D:/data/YinChuan/1h_analysis/'; ls = os.listdir(path)
    
    # ----拼接目录中文件
    df = pd.read_excel(path + ls[0], index_col = 0)
    for f in ls[:50]:
        print(f)
        fpath = path + f
        df = pd.concat([df, pd.read_excel(fpath, index_col = 0)])
    
    # ----画图
    df1 = df.loc[(df['hourly rainfall'] >= 0.1) & (df['1h_zr'] >= 0.1) & (df['1h_mlp'] >= 0.1)]
    

    
    t = df1['hourly rainfall'].values
    p1 = df1['1h_zr'].values
    p2 = df1['1h_mlp'].values
    mytools.tp_scatter(t, p1, xylabel = '1h rainfall (mm)', minmax = [0.1, 20],title='300zr',stat=True)
    mytools.tp_scatter(t, p2, xylabel = '1h rainfall (mm)', minmax = [0.1, 20],title='mlp',stat=True)
