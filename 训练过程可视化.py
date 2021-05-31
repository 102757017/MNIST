from selenium import webdriver
import os
import sys


exe=os.path.join(sys.prefix,"Scripts","tensorboard.exe")
obj=os.path.join(os.path.dirname(__file__),"log")
cmd="\""+exe+"\""+" --logdir="+obj
print("command命令是："+ cmd)


options = webdriver.ChromeOptions()
options.binary_location = r'D:\Program Files\local-chromium\575458\chrome-win32\chrome.exe'
browser = webdriver.Chrome(chrome_options=options)
browser.get('localhost:6006')

os.system(cmd) #执行command

