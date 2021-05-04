# coding: utf-8
#!/usr/bin/python
import numpy as np
from keras.models import load_model
from keras.models import Model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import datetime

f=np.load('mnist.npz')
x_train = f['x_train']
y_train = f['y_train']

#因为卷积层要求输入的input_shape=(28,28,1)，因此需要将输入数据增加一个维度，变成60000个(28,28,1)的数组
x_train=x_train.reshape(x_train.shape[0],28,28,1)

print('x_train',x_train.shape)
print('y_train',y_train.shape)


#加载模型
model = load_model('model.h5')
#删除最后1层
model = Model(inputs=model.input, outputs=model.get_layer('dropout_2').output)
#预测结果，返回10个数据的一维数组，类型为浮点数，每个数表示结果为该类的概率
#使用predict时,必须设置batch_size,否则否则PCI总线之间的数据传输次数过多，性能会非常低下
#不同的batch_size，得到的预测结果不一样，原因是因为batch normalize 时用的是被预测的x的均值，而每一批x的值是不一样的，所以结果会随batch_size的改变而改变
#想要同一个图片的预测概率不变，只能不用batch_size
y_predict=model.predict(x_train,batch_size=32,verbose=1)
print("y_predict.shape:",y_predict.shape)


#PCA是线性降维方法，PCA缺省参数为None，所有特征被保留，此处降为3维
t=datetime.datetime.now()
X_pca = PCA(3).fit_transform(x_train.reshape(x_train.shape[0],-1))
X_pca2 = PCA(3).fit_transform(y_predict)
t2=datetime.datetime.now()-t
print("PCA降维耗时:",t2)
print("x_train PCA降维后:",X_pca.shape)
print("y_predict PCA降维后:",X_pca2.shape)

# 生成画布
fig = plt.figure()
#生成子图，将画布分割成1行2列，图像画在从左到右从上到下的第2块
ax1=fig.add_subplot(121,projection='3d')
#使用PCA降维绘制3D散点图
ax1.scatter3D(X_pca[:, 0], X_pca[:, 1],X_pca[:, 2],s=1,c=y_train)

ax2=fig.add_subplot(122,projection='3d')
#使用PCA降维绘制3D散点图
ax2.scatter3D(X_pca2[:, 0], X_pca2[:, 1],X_pca2[:, 2],s=1,c=y_train)


plt.show()