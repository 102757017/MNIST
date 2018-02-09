
# coding: utf-8

#!/usr/bin/python

import numpy as np
import time
from PIL import Image

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

for index, x in enumerate(x_train):
    print(index)
    im=Image.fromarray(x)
    im.save(str(index)+".jpg")
    f = open('list.txt','a') # 追加模式
    f.write(str(index)+".jpg "+ str(y_train[index]) + '\n')
