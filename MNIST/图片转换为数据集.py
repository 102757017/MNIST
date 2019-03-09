# coding: utf-8
#!/usr/bin/python
import numpy as np
from PIL import Image


f=open("list.txt","r")
lines=f.read().split("\n")
#uint8格式只能表示0~255,占用存储空间和内存空间比较小
data_imgs=np.zeros((501,28,28), dtype=np.uint8)
data_ans=np.zeros((501,1), dtype=np.uint8)
for index,line in enumerate(lines):
    
    file=line.split(" ")[0]
    char=line.split(" ")[1]
    im=Image.open(file)
    a=np.array(im)
    data_imgs[index,:,:]=a
    data_ans[index,:]=char
print(data_imgs.shape)
np.savez('mydataset',x_train=data_imgs,y_train=data_ans)
