# coding: utf-8
#!/usr/bin/python
import numpy as np
from keras.callbacks import TensorBoard
#随机数种子不变的情况下，random.random()生成的随机数是不变的

np.random.seed(123)

from keras.layers import Dense #导入全连接神经层

from keras.layers import Dropout #导入正则化，Dropout将在训练过程中每次更新参数时按一定概率(rate)随机断开输入神经元

from keras.layers import Conv2D #导入卷积层

from keras.layers import MaxPooling2D #导入池化层

from keras.layers import Flatten

from keras.layers import Input #导入输入数据层

from keras.layers import LeakyReLU #导入激活函数层

from keras.layers import BatchNormalization #导入BN层

from keras.models import Model #导入函数式模型

from keras.utils import np_utils #数据预处理为0~1

from keras.models import load_model 

from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint #训练途中自动保存模型
from keras.callbacks import EarlyStopping
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
x_test=x_test.astype("float32")

#将输入数组中的数据转换为0~1之间的数
x_train /= 255

#y_train此时为一维数组
print('y_train',y_train.shape)
print(y_train[0:5])


#函数式模型
inputs = Input(shape=(28,28,1))
#kernel_initializer：应该根据激活函数选择权重初始化方法，不合适的初始化方法会导致dead relu，会造成梯度消失，网络不收敛
#未指定kernel_initializer的情况下，进行二次训练时权重不好复位，dead relu也不会恢复。
#激活函数为tanh时使用Xavier初始化较好，激活函数为relu时，使用he_normal、he_uniform、MSRA较好
x=Conv2D(32,(3,3),kernel_initializer='he_uniform')(inputs)

#当深层网络难以收敛时，可以使用BN层加快收敛，使用BN层可以避免dead ReLU（使用relu运算速度较快），使用BN层时可以取消dropout层
#x=BatchNormalization()(x)

#激活函数使用LeakyReLU，当不激活时，LeakyReLU仍然会有非零输出值，使dead relu有复活的机会
#分类问题对数据分布的变换不敏感，使用BN层效果较好，对于非分类问题要慎用BN，如GAN，由于改变了数据的分布，可能导致精度降低
x=LeakyReLU(alpha=0.2)(x)
x=Conv2D(32,(3,3),kernel_initializer='he_uniform')(x)
#x=BatchNormalization()(x)
x=LeakyReLU(alpha=0.2)(x)
x=MaxPooling2D(pool_size=(2,2))(x)
x=Dropout(0.25)(x)
x=Flatten()(x)
x=Dense(128,activation='relu',kernel_initializer='he_uniform')(x)
x=Dropout(0.5)(x)
predictions=Dense(10,activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)

#编译模型时, 我们需要声明损失函数和优化器 (SGD, Adam 等等)
#optimizer：优化器，该参数可指定为已预定义的优化器名，如rmsprop、adagrad、adam
#loss：损失函数,该参数为模型试图最小化的目标函数，它可为预定义的损失函数名，如categorical_crossentropy、mse被叫做均方误差,MAE为绝对误差
#如果你的 targets 是 one-hot 编码，用 categorical_crossentropy  one-hot 编码：[0, 0, 1], [1, 0, 0], [0, 1, 0]
#如果你的 tagets 是 数字编码 ，用 sparse_categorical_crossentropy  数字编码：2, 0, 1
#metrics：指标列表,对分类问题，数字编码时一般将该列表设置为metrics=['sparse_categorical_accuracy']
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['sparse_categorical_accuracy'])



filepath = "weights-improvement.hdf5"
# 每个epoch确认确认monitor的值，如果训练效果提升, 则将权重保存, 每提升一次, 保存一次
#mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，例如，当监测值为val_acc时，模式应为max，当监测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True,mode='auto')

#如果val_loss相比上个epoch没有下降，则经过patience个epoch后停止训练
early_stop=EarlyStopping(monitor='val_loss', verbose=0, patience=5,mode='auto')

#TensorBoard的回调函数
tb = TensorBoard(log_dir="./log",  # 日志文件保存位置
                histogram_freq=1,  # 对于模型中各个层计算激活值和模型权重直方图的频率（训练轮数中）。 如果设置成 0 ，直方图不会被计算。对于直方图可视化的验证数据（或分离数据）一定要明确的指出。
                batch_size=32,     # 用多大量的数据计算直方图
                write_graph=False,    #是否在 TensorBoard 中可视化图像。 如果 write_graph 被设置为 True，日志文件会变得非常大。
                write_grads=True,    # 是否在tensorboard中可视化梯度直方图
                write_images=False,   # 是否在tensorboard中以图像形式可视化模型权重
                update_freq='batch')   # 更新频率,batch或epoch


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
#validation_split：用作验证集的训练数据的比例。 模型将分出一部分不会被训练的验证数据，也可以直接指定validation_data=(x_val,y_val)
history =model.fit(x_train,y_train,batch_size=100,epochs=1,verbose=1,callbacks=[checkpoint,early_stop,tb],validation_split=0.1)
#返回记录字典，包括每一次迭代的训练误差率和验证误差率


# 保存模型
model.save('model.h5')   # HDF5文件，pip install h5py

# 评估模型
#model.evaluate返回的是一个list,其中第一个元素为loss指标，其它元素为metrias中定义的指标，metrias指定了N个指标则返回N个元素
loss,accuracy = model.evaluate(x_test,y_test,batch_size=1000)
print('\ntest loss',loss)
print('accuracy',accuracy)


#加载模型
model = load_model('model.h5')

#预测结果，返回10个数据的一维数组，类型为浮点数，每个数表示结果为该类的概率
#使用predict时,必须设置batch_size,否则否则PCI总线之间的数据传输次数过多，性能会非常低下
#不同的batch_size，得到的预测结果不一样，原因是因为batch normalize 时用的是被预测的x的均值，而每一批x的值是不一样的，所以结果会随batch_size的改变而改变
#想要同一个图片的预测概率不变，只能不用batch_size
y_predict=model.predict(x_test[0:5],batch_size=32,verbose=1)
