from selenium import webdriver
import os
import sys



exe=os.path.join(sys.prefix,"Scripts","tensorboard.exe")
obj=os.path.join(os.path.dirname(__file__),"log")
cmd="\""+exe+"\""+" --logdir="+obj
print("command命令是："+ cmd)

browser = webdriver.Chrome()
browser.get('localhost:6006')

os.system(cmd) #执行command

