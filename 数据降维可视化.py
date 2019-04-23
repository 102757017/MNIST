# -*- coding: UTF-8 -*-
#from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from tsne import bh_sne as TSNE

f=np.load('mnist.npz')
x_test = f['x_test']
y_test=f["y_test"]
print(x_test.shape,x_test.dtype)

x_test=x_test.reshape(10000,28*28)
x_test=x_test.astype(float)
#TSNE与keras中的池化层作用类似，是非线性降维，PCA是线性降维
#TSNE期望的输入数据是np.array，会将数据降成两维,，
#T-SNE，PCA是无监督学习方法，输入的数据不需要标记，根据样本间的相似性对样本集进行自动分类
X_tsne = TSNE(x_test)

#PCA缺省参数为None，所有特征被保留
X_pca = PCA(2).fit_transform(x_test)

print("T-SNE降维后:",X_tsne.shape)
print("PCA降维后:",X_pca.shape)


#使用T-SNE降维绘子图
plt.subplot(121)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_test,label="T-SNE")

#使用PCA降维绘子图
plt.subplot(122)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test,label="PCA")


plt.show()
