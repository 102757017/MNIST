# coding: utf-8
#!/usr/bin/python
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
#随机数种子不变的情况下，random.random()生成的随机数是不变的

np.random.seed(123)

from tensorflow.keras.layers import Dense #导入全连接神经层

from tensorflow.keras.layers import Dropout #导入正则化，Dropout将在训练过程中每次更新参数时按一定概率(rate)随机断开输入神经元

from tensorflow.keras.layers import Conv2D #导入卷积层

from tensorflow.keras.layers import MaxPooling2D #导入池化层

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Input #导入输入数据层

from tensorflow.keras.layers import LeakyReLU #导入激活函数层

from tensorflow.keras.layers import BatchNormalization #导入BN层

from tensorflow.keras.layers import Add

from tensorflow.keras.models import Model #导入函数式模型

from tensorflow.keras.utils import to_categorical #数据预处理为0~1

from tensorflow.keras.models import load_model 

from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint #训练途中自动保存模型
from tensorflow.keras.callbacks import EarlyStopping
import os
from tensorflow.keras.utils import plot_model
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


#将一维数组转换为分类问题，0→[1,0,0,0,0,0,0,0,0,0]  1→[0,1,0,0,0,0,0,0,0,0]依此类推
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

#y_train此时为二维数组
print('y_train',y_train.shape)
print(y_train[0:5])


def res_block(x,kernel_size):
    #1*1的卷积用来降维/升维，strides=(2,2)的作用为改变shape，使其与pool后的shape一致
    b_x=Conv2D(kernel_size,(1,1),strides=(2,2),kernel_initializer='he_uniform')(x)
    
    x=Conv2D(kernel_size,(3,3),padding="same",kernel_initializer='he_uniform')(x)
    x=BatchNormalization()(x)
    x=LeakyReLU(alpha=0.2)(x)
    
    x=Conv2D(kernel_size,(3,3),padding="same",kernel_initializer='he_uniform')(x)    
    x=BatchNormalization()(x)
     
    x=MaxPooling2D(2,2)(x)
    
    x=Add()([b_x,x])
    x=LeakyReLU(alpha=0.2)(x)
    return x

#函数式模型
inputs = Input(shape=(28,28,1))
x=inputs

for i in range(2):
    x=res_block(x,kernel_size=(i+1)*32)
    
#降维减少后面全连接层的计算量
x=Conv2D(10,(1,1),kernel_initializer='he_uniform')(x)    
x=Flatten()(x)
predictions=Dense(10,activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)

#编译模型时, 我们需要声明损失函数和优化器 (SGD, Adam 等等)
#optimizer：优化器，该参数可指定为已预定义的优化器名，如rmsprop、adagrad、adam
#loss：损失函数,该参数为模型试图最小化的目标函数，它可为预定义的损失函数名，如categorical_crossentropy、mse被叫做均方误差,MAE为绝对误差
#metrics：指标列表,对分类问题，我们一般将该列表设置为metrics=['accuracy']
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#给系统添加环境变量，修改的环境变量是临时改变的，当程序停止时修改的环境变量失效（系统变量不会改变）
os.environ["Path"] += os.pathsep + r"D:\Program Files\WinPython-64bit-3.6.1.0Qt5\graphviz\bin"
plot_model(model, to_file='resnet.png',show_shapes=True)

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
history =model.fit(x_train,y_train,batch_size=100,epochs=1,verbose=1,callbacks=[checkpoint,early_stop],validation_split=0.1)
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
plt.plot(history.history['accuracy'], label='train_acc')
leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.show()

#加载模型
model = load_model('model.h5')

#预测结果，返回10个数据的一维数组，类型为浮点数，每个数表示结果为该类的概率
#使用predict时,必须设置batch_size,否则否则PCI总线之间的数据传输次数过多，性能会非常低下
#不同的batch_size，得到的预测结果不一样，原因是因为batch normalize 时用的是被预测的x的均值，而每一批x的值是不一样的，所以结果会随batch_size的改变而改变
#想要同一个图片的预测概率不变，只能不用batch_size
y_predict=model.predict(x_test[0:10],batch_size=32,verbose=1)
