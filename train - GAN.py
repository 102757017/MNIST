# coding: utf-8
#!/usr/bin/python
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense #导入全连接神经层

from tensorflow.keras.layers import Dropout #导入正则化，Dropout将在训练过程中每次更新参数时按一定概率(rate)随机断开输入神经元

from tensorflow.keras.layers import Conv2D #导入卷积层

from tensorflow.keras.layers import MaxPooling2D #导入池化层

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Input #导入输入数据层

from tensorflow.keras.layers import LeakyReLU #导入激活函数层

from tensorflow.keras.layers import BatchNormalization #导入BN层

from tensorflow.keras.layers import Reshape

from tensorflow.keras.layers import Add

from tensorflow.keras.models import Model #导入函数式模型

from tensorflow.keras.utils import to_categorical #数据预处理为0~1

from tensorflow.keras.models import load_model 

from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint #训练途中自动保存模型
from tensorflow.keras.callbacks import EarlyStopping
import os
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam

#给系统添加环境变量，修改的环境变量是临时改变的，当程序停止时修改的环境变量失效（系统变量不会改变）
os.environ["Path"] += os.pathsep + r"D:\Program Files\WinPython-64bit-3.6.1.0Qt5\graphviz\bin"


(x_train,y_train), (x_test,y_test) = mnist.load_data()
print('x_train',x_train.shape)
print('y_train',y_train.shape)
print('x_test',x_test.shape)
print('y_test',y_test.shape)


#因为卷积层要求输入的input_shape=(28,28,1)，因此需要将输入数据增加一个维度，变成60000个(28,28,1)的数组
x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)

#将数据转换为浮点数
x_train=x_train.astype('float32')
x_test=x_test.astype("float32")




#因为生成网络的最后一层需要使用tanh作为激活函数，因此需要将输入数组中的数据转换为-1~1之间的数
x_train=x_train/127.5-1
x_test=x_test/127.5-1

#y_train此时为一维数组
print('y_train',y_train.shape)
print(y_train[0:5])


#将一维数组转换为分类问题，0→[1,0,0,0,0,0,0,0,0,0]  1→[0,1,0,0,0,0,0,0,0,0]依此类推
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

#y_train此时为二维数组
print('y_train',y_train.shape)
print(y_train[0:5])



#输入为图片，输出为[0,1]，表示真假
def discriminator(inputs = Input(shape=(28,28,1))): 
  x=inputs
  x=Flatten()(x)
  x=Dense(512)(x)
  x=LeakyReLU(alpha=0.2)(x)
  x=Dropout(0.4)(x)
  x=Dense(256)(x)
  x=LeakyReLU(alpha=0.2)(x)
  x=Dense(1,activation='sigmoid')(x)
  return x

#输入为100维的随机数（0~1），输出为图片
def generator(inputs = Input(shape=(100,))):
  x=inputs
  
  x=Dense(256)(x)
  x=LeakyReLU(alpha=0.2)(x)
  x=BatchNormalization()(x)
  x=Dense(512)(x)
  x=LeakyReLU(alpha=0.2)(x)
  x=BatchNormalization()(x)
  x=Dense(1024)(x)
  x=LeakyReLU(alpha=0.2)(x)
  x=BatchNormalization()(x)
  
  x=Dense(784,activation='tanh')(x)
  x=Reshape((28,28,1),name="gen_img")(x)
  return x



#定义判别模型
inputs = Input(shape=(28,28,1))
output_d=discriminator(inputs)
discriminator_model=Model(inputs=inputs, outputs=output_d)
#必须设置Adam的参数，默认参数可能会跑飞
discriminator_model.compile(loss='binary_crossentropy',optimizer=Adam(0.0002, 0.5),metrics=['accuracy'])
plot_model(discriminator_model, to_file='discriminator.png',show_shapes=True)



#定义生成模型
inputs2 = Input(shape=(100,))
#输出生成的图像
output_g=generator(inputs2)
#将生成的图像输给判别网络，此处的判别网络和上面的判别网络是共用权重的,但是训练生成模型时需要将discriminator_model的权重冻结。
output_d2=discriminator_model(output_g)
generator_model=Model(inputs=inputs2, outputs=output_d2)
#将discriminator_model的权重冻结
discriminator_model.trainable=False
#冻结权重后重新编译模型，冻结才能生效，因此前面的discriminator_model的权重是未冻结的，generator_model中使用到的generator_model的权重被冻结了
generator_model.compile(loss='binary_crossentropy',optimizer=Adam(0.0002, 0.5))
plot_model(generator_model, to_file='generator.png',show_shapes=True)

#创建一个新model, 使得它的输出(outputs)是generator_model 中output_g的输出
model_mid = Model(inputs=generator_model.input, outputs=generator_model.get_layer('gen_img').output)



for epoch in range(10000):
  #在训练集中随机抽取batch_size=128张图片，x_train.shape[0]=60000
  idx = np.random.randint(0, x_train.shape[0], 128)
  imgs = x_train[idx]
  
  #随机生成batch_size=128组随机数
  noise = np.random.normal(0, 1, (128, 100))
  gen_imgs = model_mid.predict(noise)
  
  
  #训练判别模型识别真实图片为真，因此标签全部为[1]
  d_loss_real = discriminator_model.train_on_batch(imgs,np.ones((128, 1), dtype=float))
  #训练判别模型识别生成的图片为假，因此标签全部为[0]
  d_loss_fake = discriminator_model.train_on_batch(gen_imgs, np.zeros((128, 1), dtype=float))
  d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
  print("d_loss_real:",d_loss_real,"d_loss_fake",d_loss_fake)
  

  #训练生成模型
  #判别模型的权重被冻结了，此时需要强迫判别模型将图片认定为真[1]，因此生成模型的权重被训练了以适应该要求。
  noise = np.random.normal(0, 1, (128, 100))
  g_loss = generator_model.train_on_batch(noise, np.ones((128, 1), dtype=float))
  print("g_loss",g_loss)
  
  
  def sample_images(epoch):
    r=5
    c=5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = model_mid.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
      for j in range(c):
        axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
        axs[i,j].axis('off')
        cnt += 1
      fig.savefig(r"images/{}.png".format(epoch))
      plt.close()
  
  #每200epoch保存一次生成器生成的图片
  if epoch % 200 == 0:
    sample_images(epoch)

d_loss_fake
