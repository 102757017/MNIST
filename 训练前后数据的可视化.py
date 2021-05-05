# coding: utf-8
#!/usr/bin/python
import numpy as np
from keras.models import load_model
from keras.models import Model
from sklearn.decomposition import PCA
import datetime
import plotly.graph_objs as go
from plotly.subplots import make_subplots


f=np.load('mnist.npz')
x_train = f['x_train']
y_train = f['y_train']

#因为卷积层要求输入的input_shape=(28,28,1)，因此需要将输入数据增加一个维度，变成60000个(28,28,1)的数组
x_train=x_train.reshape(x_train.shape[0],28,28,1)

print('x_train',x_train.shape)
print('y_train',y_train.shape)

#将数据转换为浮点数
x_train=x_train.astype("float32")
y_train=y_train.astype("float32")

#将输入数组中的数据转换为0~1之间的数
x_train /= 255

#加载模型
model = load_model("model.h5")
#删除最后1层
model = Model(inputs=model.input, outputs=model.get_layer('dropout_2').output)
#预测结果，返回10个数据的一维数组，类型为浮点数，每个数表示结果为该类的概率
#使用predict时,必须设置batch_size,否则否则PCI总线之间的数据传输次数过多，性能会非常低下
#不同的batch_size，得到的预测结果不一样，原因是因为batch normalize 时用的是被预测的x的均值，而每一批x的值是不一样的，所以结果会随batch_size的改变而改变
#想要同一个图片的预测概率不变，只能不用batch_size
y_predict=model.predict(x_train,batch_size=1024,verbose=1)
print("y_predict.shape:",y_predict.shape)


#PCA是线性降维方法，PCA缺省参数为None，所有特征被保留，此处降为3维
t=datetime.datetime.now()
X_pca = PCA(3).fit_transform(x_train.reshape(x_train.shape[0],-1))
X_pca2 = PCA(3).fit_transform(y_predict)
t2=datetime.datetime.now()-t
print("PCA降维耗时:",t2)
print("x_train PCA降维后:",X_pca.shape)
print("y_predict PCA降维后:",X_pca2.shape)


#子图1
trace0 = go.Scatter3d(x = X_pca[:, 0], y = X_pca[:, 1], z = X_pca[:, 2],mode = 'markers', marker = dict(size = 1,color = y_train, colorscale = 'Viridis'))
#子图2
trace1 = go.Scatter3d(x = X_pca2[:, 0], y = X_pca2[:, 1], z = X_pca2[:, 2],mode = 'markers', marker = dict(size = 1,color = y_train, colorscale = 'Viridis'))

'''
specs用来指定子图的类型
“ xy”：散点图，柱状图等的2D子图类型。如果未指定类型，则这是默认设置。
“scene”：用于3D散点图，圆锥体等的子图。
“polar”：极坐标散点图，极坐标上的柱状图等。
“ternary”：三元图子图。
“ mapbox”：地理地图子图。
'''
fig = make_subplots(rows=1,cols=2,subplot_titles=["原始数据PCA降维", "训练后的数据（去掉最后一层）PCA降维"],specs=[[{"type": "scene"}, {"type": "scene"}]])
fig.append_trace(trace0, 1, 1) 
fig.append_trace(trace1, 1, 2) 
fig.show()