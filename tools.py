import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import time
def tp_eval(x,y):
    
    d = (y-x)
    bias = np.sum(abs(d))/np.sum(x)*100
    me = np.mean(d)
    # mae = np.mean(abs(d))
    rmse = (np.mean(d**2))**0.5
    cc = np.corrcoef(x.flatten(), y.flatten())[0,1]
    # return me, mae, rmse, cc
    return bias, me, rmse, cc

from scipy.optimize import curve_fit
# def function(x, k, b):
#     return k*x+b
def function(x, k):
    return k*x
def tp_scatter(x,y,label=None,
               xylabel=None,title=None,minmax=None,
               num=None,stat=None):
    '''
    num: 图序号
    stat：统计参数
    '''
    # set rcparams
    plt.rcParams['figure.figsize']=(20,16)
    plt.rcParams['figure.dpi']=600
    plt.rcParams['font.size']=30
    plt.rcParams['font.serif']='Times New Roman'
    
    
    x = x.flatten()
    y = y.flatten()
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    z1 = z*len(z)
    
    # draw
    fig = plt.figure(figsize=(14,14),dpi=600)
    ax = fig.add_subplot(111)
    scatter = ax.scatter(x, y, label=label, c=z, cmap='Spectral_r')
    if label != None:
        plt.legend(loc='upper right')
    fig.colorbar(scatter)

    
    ax.set_xlabel(xylabel+(' (obs)'))
    ax.set_ylabel(xylabel)
    plt.title(title)
    if minmax:
        ax.plot(minmax,minmax,c='black')
        ax.set_ylim(minmax[0],minmax[1])
        ax.set_xlim(minmax[0],minmax[1])
    
    if num:
        ax.text(1,0.95*minmax[1],num)

    if stat:
        ax.text(1,0.90*minmax[1],'points='+str(stat[0]))
        ax.text(1,0.85*minmax[1],'BIAS='+str(stat[1])[:5]+"%")
        ax.text(1,0.80*minmax[1],'ME='+str(stat[2])[:5])
        ax.text(1,0.75*minmax[1],'RMSE='+str(stat[3])[:5])
        ax.text(1,0.70*minmax[1],'CORR='+str(stat[4])[:5])
    
    # print('在拟合了')
    # p_est, err_est = curve_fit(function, x, y)
    # # ax.plot(x, function(x, p_est[0], p_est[1]), linestyle=':')
    # ax.plot(x, function(x, p_est[0]), linestyle=':')
    # print('拟合完了')
    
    
    # ax.set_xscale("log", base = 10)
    # ax.set_yscale("log", base = 10)
    
    plt.grid()
    
    
    # ts = time.time()
    # ts = time.localtime(ts)
    # ts = time.strftime("%Y-%m-%d %H%M%S", ts)
    # plt.savefig('e:/'+ts+'.png',dpi=600)
    plt.show()

def cm_scatter(x,y,label=None,
               xlabel=None, ylabel=None, title=None,xminmax=None,yminmax=None,
               num=None,stat=None):
    '''
    num: 图序号
    stat：统计参数
    '''
    # set rcparams
    plt.rcParams['figure.figsize']=(10,8)
    plt.rcParams['figure.dpi']=400
    plt.rcParams['font.size']=20
    plt.rcParams['font.serif']='Times New Roman'
    
    
    x = x.flatten()
    y = y.flatten()
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    # z1 = z*len(z)
    
    # draw
    fig = plt.figure(figsize=(10,10),dpi=600)
    ax = fig.add_subplot(111)
    scatter = ax.scatter(x, y, label=label, c=z, cmap='Spectral_r')
    if label != None:
        plt.legend(loc='upper right')
    fig.colorbar(scatter)

    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)
    if yminmax != None and xminmax != None:
        ax.set_ylim(yminmax[0],yminmax[1])
        ax.set_xlim(xminmax[0],xminmax[1])
    
    if num:
        ax.text(1,0.95*yminmax[1],num)

    if stat:
        ax.text(1,90*yminmax[1],'points='+str(stat[0]))
        ax.text(1,85*yminmax[1],'MAE='+str(stat[1])[:5])
        ax.text(1,80*yminmax[1],'RMSE='+str(stat[2])[:5])
        ax.text(1,75*yminmax[1],'CORR='+str(stat[3])[:5])
    
    plt.grid()
    plt.show()


def plot(x, y, labels=None,xlabel=None,ylabel=None,text=None, p=None):
    plt.rcParams['figure.figsize']=(10,8)
    plt.rcParams['figure.dpi']=400
    plt.rcParams['font.size']=20
    plt.rcParams['font.serif']='Times New Roman'
    
    plt.plot(y,label=labels,marker = 'o')
    plt.xticks(ticks = np.arange(len(y)),labels = x)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if text != None:
        plt.text(p[0], p[1], text)
    plt.show()
    # plt.grid()
    
# path = "E:\\【paper】\\work\\dsd\\select_dsd_radar.csv"
# import pandas as pd
# df = pd.read_csv(path)
# t = df['zh'].values
# p0 = df['zh.1'].values
# p = df['zh2'].values

# import numpy as np
# loc = np.where((t>0) & (p>0))
# t = t[loc]
# p0 = p0[loc]
# p=p[loc]

# d = p0-t
# stat = [len(d), np.mean(d), np.mean(abs(d)), (np.mean(d**2))**0.5, np.corrcoef(t, p0)[0,1]]
# tp_scatter(t,p0,
#                xylabel='$Z_{H}$',minmax=[0,80],
#                num='(a)',stat=stat)
# d = p-t
# stat = [len(d), np.mean(d), np.mean(abs(d)), (np.mean(d**2))**0.5, np.corrcoef(t, p)[0,1]]
# tp_scatter(t,p,
#                xylabel='$Z_{H}$',minmax=[0,80],
#                num='(b)',stat=stat)
if __name__ == '__main__':
    import pandas as pd
    df = pd.read_excel('history/2023-05-27_zs_resample/sheet1的副本.xlsx',sheet_name = 'Sheet1').iloc[:,1:].values
    plot(['0-10','10-25','25-50','50-'],df,
         ['MLP', '$Z=238R^{1.57}$','$Z=300R^{1.4}$','$Z=200R^{1.6}$'],
         'rain rate interval (mm/h)', 'ME','(a)',[0,-20])
    df = pd.read_excel('history/2023-05-27_zs_resample/sheet1的副本.xlsx',sheet_name = 'Sheet2').iloc[:,1:].values
    plot(['0-10','10-25','25-50','50-'],df,
         ['MLP', '$Z=238R^{1.57}$','$Z=300R^{1.4}$','$Z=200R^{1.6}$'],
         'rain rate interval (mm/h)', 'RMSE','(b)',[0,30])
