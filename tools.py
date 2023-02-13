import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

def plot(x, y, label=None, stat=None,num=None , 
         xminmax=None, yminmax=None,xlabel=None, ylabel=None, line=0):
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
    z1 = z*len(z)
    
    # draw
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(x, y, label=label, c=z1, cmap='Spectral_r')
    
    if line == 1:
        ax.plot(xminmax,yminmax,c='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if yminmax:
        ax.set_ylim(0,100)
    if xminmax:
        ax.set_xlim(0,100)
    if label:
        plt.legend(loc='upper right')
    
    ax.text(1,95,num)

    if stat:
        ax.text(1,90,'points='+str(stat[0]))
        ax.text(1,85,'MAE='+str(stat[1])[:5])
        ax.text(1,80,'RMSE='+str(stat[2])[:5])
        ax.text(1,75,'CORR='+str(stat[3])[:5])
    fig.colorbar(scatter)
    plt.show()