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

from keras.models import load_model 

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

#因为卷积层要求输入的input_shape=(28,28,1)，因此需要将输入数据增加一个维度，变成60000个(28,28,1)的数组
x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)

#将数据转换为浮点数
x_train=x_train.astype('float32')
y_train=y_train.astype("float32")
x_test=x_test.astype("float32")
y_test=y_test.astype("float32")

#将输入数组中的数据转换为0~1之间的数
x_train /= 255
y_test /= 255

#y_train此时为一维数组
print('y_train',y_train.shape)
print(y_train[0:5])


#将一维数组转换为分类问题，0→[1,0,0,0,0,0,0,0,0,0]  1→[0,1,0,0,0,0,0,0,0,0]依此类推
y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)

#y_train此时为二维数组
print('y_train',y_train.shape)
print(y_train[0:5])




#加载模型
model = load_model('model.h5')
print(model.summary())



#预测结果，返回10个数据的一维数组，类型为浮点数，每个数表示结果为该类的概率
#使用predict时,必须设置batch_size,否则否则PCI总线之间的数据传输次数过多，性能会非常低下
#不同的batch_size，得到的预测结果不一样，原因是因为batch normalize 时用的是被预测的x的均值，而每一批x的值是不一样的，所以结果会随batch_size的改变而改变
#想要同一个图片的预测概率不变，只能不用batch_size
y_predict=model.predict(x_test,batch_size=1000,verbose=2)
print(y_predict)

#对独热编码进行解码
predict=np.argmax(y_predict, axis=1)
print(predict)
