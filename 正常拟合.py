import pandas as pd
import numpy as np

# # 多高度
# path = 'result/'
# x_train = np.load(path+'x_train.npy')[:,0:]#.reshape(-1,1)
# x_vali = np.load(path+'x_vali.npy')[:,0:]#.reshape(-1,1)
# x_test = np.load(path+'x_test.npy')[:,0:]#.reshape(-1,1)
# y_train = np.load(path+'y_train.npy').reshape(-1,1)
# y_vali = np.load(path+'y_vali.npy').reshape(-1,1)
# y_test = np.load(path+'y_test.npy').reshape(-1,1)

# x = np.vstack([x_train, x_vali])
# x = np.vstack([x, x_test])
# y = np.vstack([y_train, y_vali])
# y = np.vstack([y, y_test])
# x = x[:,1]
# y = y[:,0]


#组合反射率
path = 'result/'
x_train = np.load(path+'x_train.npy')[:,0:]#.reshape(-1,1)
x_vali = np.load(path+'x_vali.npy')[:,0:]#.reshape(-1,1)
x_test = np.load(path+'x_test.npy')[:,0:]#.reshape(-1,1)
y_train = np.load(path+'y_train.npy').reshape(-1,1)
y_vali = np.load(path+'y_vali.npy').reshape(-1,1)
y_test = np.load(path+'y_test.npy').reshape(-1,1)

x = np.vstack([x_train, x_vali])
x = np.vstack([x, x_test])
y = np.vstack([y_train, y_vali])
y = np.vstack([y, y_test])
x = x[:,0]
y = y[:,0]



from scipy.optimize import curve_fit
def function(x, a, b):
    return a*(10**(x/10))**b
p, err = curve_fit(function, x, y)
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.scatter(x, function(x, p[0], p[1]))
