
# coding: utf-8

#!/usr/bin/python

import numpy as np



#随机数种子不变的情况下，random.random()生成的随机数是不变的


np.random.seed(123)

from keras.models import Sequential

from keras.layers import Dense #导入全连接神经层

from keras.layers import Dropout #导入正则化，Dropout将在训练过程中每次更新参数时按一定概率(rate)随机断开输入神经元

from keras.layers import Activation #导入激活函数

from keras.layers import Convolution2D #导入卷积层

from keras.layers import MaxPooling2D #导入池化层

from keras.layers import Flatten

from keras.utils import np_utils #数据预处理为0~1

from keras.datasets import mnist #导入手写数据集

from matplotlib import pyplot as plt

f=np.load('mnist.npz')
print(type(f),f)
x_train = f['x_train']
y_train = f['y_train']
x_test = f['x_test']
y_test = f['y_test'] 
print('x_train',x_train.shape)
print('y_train',y_train.shape)
print('x_test',x_test.shape)
print('y_test',y_test.shape)


import matplotlib
plt.imshow(x_train[0])
#plt.show()

x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)

#将数据转换为浮点数
x_train=x_train.astype('float32')
y_train=y_train.astype("float32")
x_test=x_test.astype("float32")
y_test=y_test.astype("float32")

#y_train此时为一维数组
print('y_train',y_train.shape)
print(y_train[0:5])

#将一维数组转换为分类问题，0→[1,0,0,0,0,0,0,0,0,0]  1→[0,1,0,0,0,0,0,0,0,0]依此类推
y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)

#y_train此时为二维数组
print('y_train',y_train.shape)
print(y_train[0:5])


#开始定义模型架构,首先声明一个顺序模型
model=Sequential()

#加入一个二维卷积层
#卷积过滤器（卷积核）的数量:32
#卷积内核的行数：3
#卷积内核的列数：3
#激活函数为'relu'，
#当使用该层作为第一层时，应提供input_shape参数。例如input_shape = (128,128,3)代表128*128的彩色RGB图像
model.add(Convolution2D(32,3,3,activation='relu',input_shape=(28,28,1)))
print(model.output_shape)

#再次加入一个二维卷积层
model.add(Convolution2D(32,3,3,activation='relu'))

#加入一个2D池化层，MaxPooling2D 是一种减少模型参数数量的方式, 其通过在前一层上滑动一个 2*2 的滤波器, 再从这个 2*2 的滤波器的 4 个值中取最大值.
#pool_size：整数或长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，如取（2，2）将使图片在两个维度上均变为原长的一半。
model.add(MaxPooling2D(pool_size=(2,2)))

#Dropout 层将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元, 目的是防止过度拟合
#控制需要断开的神经元的比例：25%
model.add(Dropout(0.25))

#Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
model.add(Flatten())

#全连接层,第一个参数是输出的大小. Keras 会自动处理层间连接.
model.add(Dense(128,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(128,activation='softmax'))

#编译模型时, 我们需要声明损失函数和优化器 (SGD, Adam 等等)
#optimizer：优化器，该参数可指定为已预定义的优化器名，如rmsprop、adagrad
#loss：损失函数,该参数为模型试图最小化的目标函数，它可为预定义的损失函数名，如categorical_crossentropy、mse
#metrics：指标列表,对分类问题，我们一般将该列表设置为metrics=['accuracy']
model.compile(loss='categorical_crossentropy',optimizer='adam',metrias=['accuracy'])

#训练模型
#batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
#nb_epochs：整数，训练的轮数，训练数据将会被遍历nb_epoch次。Keras中nb开头的变量均为”number of”的意思
#verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
model.fit(x_train,y_train,batch_size=32,nb_epoch=10,verbose=1)
