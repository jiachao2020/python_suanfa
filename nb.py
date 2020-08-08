import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
import datetime


def date_str(days=0, str=""):
    s = datetime.date.strftime(datetime.date.today()-datetime.timedelta(days=days), "%Y-%m-%d")
    return s

def decToBin(num):
    arry = []
    while True:
        arry.append(str(num % 2))
        num = num // 2
        if num == 0:
            break
    return "".join(arry[::-1])
null = ''
url1 = "http://prod.op.stats.raipeng.cn/credit-stats/channelDataOfCreditLoan/loan/list"
url2= "http://prod.op.stats.raipeng.cn/market-stats/channelDataOfCreditLoan/loan/list"
url3= 'http://prod.op.credit.statis.raipeng.cn/channelDataOfCreditLoan/special/list'
data = {
    "startTime": date_str(),
    "endTime": date_str(),
}
headers = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
}

p={}
r = requests.post(url1, data, headers=headers)
r.encoding = 'UTF-8'
p1 = r.text.replace("\ufeff", "")
p1=eval(p1)
p1.pop('rows')
p[1]=p1

r = requests.post(url2, data, headers=headers)
r.encoding = 'UTF-8'
p2 = r.text.replace("\ufeff", "")
p2=eval(p2)
p2.pop('rows')
p[2]=p2

r = requests.post(url3, data, headers=headers)
r.encoding = 'UTF-8'
p3 = r.text.replace("\ufeff", "")
p3=eval(p3)
p3.pop('rows')
p[3]=p3

p=str(p)

b=p.encode('utf8')
b=list(b)

c,p,q=[],'',[]
for i in b:
    s=decToBin(i)
    if len(s)!=8:
        s='0'*(8-len(s))+s
    c.append(s)
print(c)
for i in c:
    p=p+i
p=list(p)
for i in p:
    if i=='1':
        q.append(255)
    else:
        q.append(0)
c=bytes(int(x,2) for x in c).decode('utf-8')
im=np.array(q)
f=im.shape[0]%1000
for i in range(1000-f):
    im=np.append(im,0)
print(im.shape)
im=im.reshape((int(im.shape[0]/1000),1000))
print(im)
im=np.pad(im,(3,3),mode='constant',constant_values=0)
print(im)
im=Image.fromarray(im)
im.show()


