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

from keras.layers import Input #导入输入数据层

from keras.models import Model #导入函数式模型

from keras.utils import np_utils #数据预处理为0~1

from keras.datasets import mnist #导入手写数据集

from keras.models import load_model 

from matplotlib import pyplot as plt

from keras.callbacks import ReduceLROnPlateau #动态调整学习率
import os

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
y_train /= 255

#y_train此时为一维数组
print('y_train',y_train.shape)
print(y_train[0:5])


#将一维数组转换为分类问题，0→[1,0,0,0,0,0,0,0,0,0]  1→[0,1,0,0,0,0,0,0,0,0]依此类推
y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)

#y_train此时为二维数组
print('y_train',y_train.shape)
print(y_train[0:5])


#函数式模型
inputs = Input(shape=(28,28,1))
w1=Convolution2D(32,3,3,activation='relu')(inputs)
w2=Convolution2D(32,3,3,activation='relu')(w1)
w3=MaxPooling2D(pool_size=(2,2))(w2)
w4=Dropout(0.25)(w3)
w5=Flatten()(w4)
w6=Dense(128,activation='relu')(w5)
w7=Dropout(0.5)(w6)
predictions=Dense(10,activation='softmax')(w7)

model = Model(inputs=inputs, outputs=predictions)

#编译模型时, 我们需要声明损失函数和优化器 (SGD, Adam 等等)
#optimizer：优化器，该参数可指定为已预定义的优化器名，如rmsprop、adagrad
#loss：损失函数,该参数为模型试图最小化的目标函数，它可为预定义的损失函数名，如categorical_crossentropy、mse被叫做均方误差,MAE为绝对误差
#metrics：指标列表,对分类问题，我们一般将该列表设置为metrics=['accuracy']
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


#学习率是每个batch权重往梯度方向下降的步距，学习率越高，loss下降越快，但是太高时会无法收敛到最优点（在附近打摆），keras默认的学习率是0.01
#设置动态学习率，使用回调函数调用
#monitor：被监测的量
#factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
#patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
#mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,patience=3, mode='auto')


filepath = "weights-improvement.hdf5"
# 每个epoch确认确认monitor的值，如果训练效果提升, 则将权重保存, 每提升一次, 保存一次
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True,mode='max')

#实现断点继续训练
if os.path.exists(filepath):
    model.load_weights(filepath)
    # 若成功加载前面保存的参数，输出下列信息
    print("checkpoint_loaded")

    
#训练模型
#batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
#nb_epochs：整数，训练的轮数，训练数据将会被遍历nb_epoch次。Keras中nb开头的变量均为”number of”的意思
#verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
#回调函数为一个list,list中可有多个回调函数,回调函数以字典logs为参数,模型的.fit()中有下列参数会被记录到logs中：
##正确率和误差，acc和loss，如果指定了验证集，还会包含验证集正确率和误差val_acc和val_loss，val_acc还额外需要在.compile中启用metrics=['accuracy']。
history =model.fit(x_train,y_train,batch_size=2000,epochs=1,verbose=1,callbacks=[reduce_lr,checkpoint])
#返回记录字典，包括每一次迭代的训练误差率和验证误差率


# 保存模型
model.save('model.h5')   # HDF5文件，pip install h5py

# 评估模型
#model.evaluate返回的是一个list,其中第一个元素为loss指标，其它元素为metrias中定义的指标，metrias指定了N个指标则返回N个元素
loss,accuracy = model.evaluate(x_test,y_test,batch_size=1000)
print('\ntest loss',loss)
print('accuracy',accuracy)



#绘图
#acc是准确率，适合于分类问题。对于回归问题，返回的准确率为0
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['acc'], label='train_acc')
leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.show()

#加载模型
model = load_model('model.h5')

#预测结果，返回10个数据的一维数组，类型为浮点数，每个数表示结果为该类的概率
#使用predict时,必须设置batch_size,否则否则PCI总线之间的数据传输次数过多，性能会非常低下
#不同的batch_size，得到的预测结果不一样，原因是因为batch normalize 时用的是被预测的x的均值，而每一批x的值是不一样的，所以结果会随batch_size的改变而改变
#想要同一个图片的预测概率不变，只能不用batch_size
y_predict=model.predict(x_test[0],batch_size=32,verbose=1)
