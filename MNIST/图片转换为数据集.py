# coding: utf-8
#!/usr/bin/python
import numpy as np
from PIL import Image


f=open("list.txt","r")
lines=f.read().split("\n")
data_imgs=np.zeros((501,28,28), dtype=float)
data_ans=np.zeros((501,1), dtype=float)
for index,line in enumerate(lines):
    
    file=line.split(" ")[0]
    char=line.split(" ")[1]
    im=Image.open(file)
    a=np.array(im)
    data_imgs[index,:,:]=a
    data_ans[index,:]=char
print(data_imgs.shape)
np.savez('mydataset',data_imgs,data_ans)
