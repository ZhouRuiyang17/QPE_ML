import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

# # 产生实验数据
# from sklearn.datasets import make_blobs
# X, y_true = make_blobs(n_samples=400, centers=4,
#                         cluster_std=0.60, random_state=0)
# X = X[:, ::-1] #交换列是为了方便画图

from sklearn.mixture import GaussianMixture as GMM
# gmm = GMM(n_components=4).fit(X)
# labels = gmm.predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
# plt.scatter(X[:, 0], y_true, c=labels, s=40, cmap='viridis');
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler, QuantileTransformer

x = np.load('x_test.npy')[:,-1].reshape(-1,1)
y = np.load('y_test.npy').reshape(-1,1)
z = np.hstack([x,y])
plt.scatter(z[:,0], z[:,1])
plt.show()

gmm = GMM(n_components=2).fit(z)
labels = gmm.predict(z)
plt.scatter(z[:,0], z[:,1], c=labels, s=40, cmap='viridis');
plt.show()
