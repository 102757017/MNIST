# coding: utf-8
#!/usr/bin/python

from keras.models import load_model 
from keras.utils import plot_model
import pydot_ng as pydot
import os


#给系统添加环境变量，修改的环境变量是临时改变的，当程序停止时修改的环境变量失效（系统变量不会改变）
os.environ["Path"] += os.pathsep + r"G:\Program Files\WinPython-64bit-3.6.1.0Qt5\graphviz\bin"


#加载模型
model = load_model('model.h5')
print(model.summary())


#pydot.Dot.create(pydot.Dot())
plot_model(model, to_file='模型结构.png',show_shapes=True)

