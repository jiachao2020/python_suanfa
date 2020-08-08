import numpy as np
from PIL import Image,ImageGrab
import re
import json
import sys
import time as time_s
from datetime import datetime, date, timedelta, time
from tkinter import *
import tkinter.messagebox
import pandas as pd
from PIL import ImageFont, Image, ImageDraw
from io import BytesIO
import win32con
import win32clipboard as wc


def time_str():
    return datetime.strftime(datetime.now(), "%H")

def date_str(days=0):
    s = date.strftime(
        date.today()-timedelta(days=days), "%Y-%m-%d")
    return s


def updt(p, d):
    print(p["channel0"])

    return (p, len(p["channel0"])+len(p["channel1"])+len(p["channel2"]))



def size(df, font):
    df_x = df.applymap(str).applymap(
        font.getsize).applymap(lambda s: s[0])
    df_y = df.applymap(str).applymap(
        font.getsize).applymap(lambda s: s[1])
    return [df_x, df_y]


def point(s):
    try:
        s = str(format(int(s), ","))
    except:
        s = ''
    return s


def table_im(p, rgb):

    heads = ["合计", "类别", "通道", "发送量", "成功量",
             "实际成功", "发送失败", "未知", "PV", "UV", "IP",
             "IP/成功量", "未知/发送量", "失败/发送量"]
    products = []
    channel0 = []
    channel1 = []
    channel2 = []
    operator = []
    totals = []
    products_len = len(p["products"])
    channel_len = len(p["channel0"])+len(p["channel1"])+len(p["channel2"])
    operator_len = len(p["operators"])
    channel0_len = len(p["channel0"])
    channel1_len = len(p["channel1"])
    channel2_len = len(p["channel2"])
    if channel0_len+channel1_len+channel2_len > 0:
        for i in range(len(p["products"])):
            products.append([
                "产品合计",
                p["products"][i]["name"],
                "",
                point(p["products"][i]["send"]),
                point(p["products"][i]["subSuccess"]),
                point(p["products"][i]["sendSuccess"]),
                point(p["products"][i]["sendFail"]),
                point(p["products"][i]["sendUnknow"]),
                point(p["products"][i]["pv"]),
                point(p["products"][i]["uv"]),
                point(p["products"][i]["ip"]),
                p["products"][i]["ipSuccess"],
                p["products"][i]["unknowSend"],
                p["products"][i]["failSend"],
            ])
        for i in range(len(p["channel0"])):
            channel0.append([
                "通道合计",
                "电信",
                p["channel0"][i]["name"],
                point(p["channel0"][i]["send"]),
                point(p["channel0"][i]["subSuccess"]),
                point(p["channel0"][i]["sendSuccess"]),
                point(p["channel0"][i]["sendFail"]),
                point(p["channel0"][i]["sendUnknow"]),
                point(p["channel0"][i]["pv"]),
                point(p["channel0"][i]["uv"]),
                point(p["channel0"][i]["ip"]),
                p["channel0"][i]["ipSuccess"],
                p["channel0"][i]["unknowSend"],
                p["channel0"][i]["failSend"],
            ])
        for i in range(len(p["channel1"])):
            channel1.append([
                "通道合计",
                "移动",
                p["channel1"][i]["name"],
                point(p["channel1"][i]["send"]),
                point(p["channel1"][i]["subSuccess"]),
                point(p["channel1"][i]["sendSuccess"]),
                point(p["channel1"][i]["sendFail"]),
                point(p["channel1"][i]["sendUnknow"]),
                point(p["channel1"][i]["pv"]),
                point(p["channel1"][i]["uv"]),
                point(p["channel1"][i]["ip"]),
                p["channel1"][i]["ipSuccess"],
                p["channel1"][i]["unknowSend"],
                p["channel1"][i]["failSend"],
            ])
        for i in range(len(p["channel2"])):
            channel2.append([
                "通道合计",
                "联通",
                p["channel2"][i]["name"],
                point(p["channel2"][i]["send"]),
                point(p["channel2"][i]["subSuccess"]),
                point(p["channel2"][i]["sendSuccess"]),
                point(p["channel2"][i]["sendFail"]),
                point(p["channel2"][i]["sendUnknow"]),
                point(p["channel2"][i]["pv"]),
                point(p["channel2"][i]["uv"]),
                point(p["channel2"][i]["ip"]),
                p["channel2"][i]["ipSuccess"],
                p["channel2"][i]["unknowSend"],
                p["channel2"][i]["failSend"],
            ])
        for i in range(len(p["operators"])):
            operator.append([
                "运营商合计",
                p["operators"][i]["name"],
                "",
                point(p["operators"][i]["send"]),
                point(p["operators"][i]["subSuccess"]),
                point(p["operators"][i]["sendSuccess"]),
                point(p["operators"][i]["sendFail"]),
                point(p["operators"][i]["sendUnknow"]),
                point(p["operators"][i]["pv"]),
                point(p["operators"][i]["uv"]),
                point(p["operators"][i]["ip"]),
                p["operators"][i]["ipSuccess"],
                p["operators"][i]["unknowSend"],
                p["operators"][i]["failSend"],
            ])
        for i in range(len(p["totals"])):
            totals.append([
                "总计",
                "",
                "",
                point(p["totals"][i]["send"]),
                point(p["totals"][i]["subSuccess"]),
                point(p["totals"][i]["sendSuccess"]),
                point(p["totals"][i]["sendFail"]),
                point(p["totals"][i]["sendUnknow"]),
                point(p["totals"][i]["pv"]),
                point(p["totals"][i]["uv"]),
                point(p["totals"][i]["ip"]),
                p["totals"][i]["ipSuccess"],
                p["totals"][i]["unknowSend"],
                p["totals"][i]["failSend"],
            ])
        df = [heads]
        df.extend(products)
        df.extend([" "*14])
        df.extend(channel0)
        df.extend(channel1)
        df.extend(channel2)
        df.extend([" "*14])
        df.extend(operator)
        df.extend([" "*14])
        df.extend(totals)
        df = pd.DataFrame(df)
        y_s, x_s = df.shape
        df_x1, df_y1 = size(df[0:1], fontbd)
        df_x2, df_y2 = size(df[1:y_s], font)
        df_x = df_x1.append(df_x2)
        df_y = df_y1.append(df_y2)
        x_d = list(df_x.max(axis=0))
        y_d = list(df_y.max(axis=1))
        x, y = [0], [0, 70]
        for i in range(len(x_d)):
            x.append(x[i]+16+x_d[i])
        for i in range(len(y_d)):
            y.append(y[i+1]+12+y_d[i])

        im1 = Image.new("RGB", (x[-1]+1, y[-1]+1), "#ffffffff")
        draw = ImageDraw.Draw(im1)
        draw.rectangle(
            (0, y[1], x[-1]-1, y[2]),
            fill=rgb
        )
        for i in range(len(x)):
            draw.rectangle(
                (x[i], 0, x[i], y[-1]),
                fill=(0, 0, 0, 0)
            )
        for i in range(len(y)):
            draw.rectangle(
                (0, y[i], x[-1], y[i]),
                fill=(0, 0, 0, 0)
            )
        for i in range(x_s):
            for j in range(y_s):
                if j == 0:
                    fon = fontbd
                else:
                    fon = font
                draw.text(
                    (
                        int(x[i]+(x[i+1]-x[i]-df_x.iloc[j, i])/2),
                        y[j+1]+5,
                    ),
                    df.iloc[j, i],
                    (0, 0, 0),
                    font=fon
                )
        draw.rectangle(
            (1, 1, x[-1]-1, y[1]-1),
            fill=rgb
        )
        draw.rectangle(
            (1, y[2]+1, x[1]-1, y[2+products_len-1]+1),
            fill=(255, 255, 255)
        )
        draw.rectangle(
            (1, y[3+products_len]+1, x[1]-1,
             y[3+products_len+channel_len-1]+1),
            fill=(255, 255, 255)
        )
        draw.rectangle(
            (1, y[4+products_len+channel_len]+1, x[1] -
             1, y[4+products_len+channel_len+operator_len-1]+1),
            fill=(255, 255, 255)
        )
        if channel0_len > 0:
            draw.rectangle(
                (x[1]+1, y[3+products_len]+1, x[2] -
                 1, y[3+products_len+channel0_len-1]+1),
                fill=(255, 255, 255)
            )
        if channel1_len > 0:
            draw.rectangle(
                (x[1]+1, y[3+products_len+channel0_len]+1, x[2] -
                 1, y[3+products_len+channel0_len+channel1_len-1]+1),
                fill=(255, 255, 255)
            )
        if channel2_len > 0:
            draw.rectangle(
                (x[1]+1, y[3+products_len+channel0_len+channel1_len]+1, x[2] -
                 1, y[3+products_len+channel0_len+channel1_len+channel2_len-1]+1),
                fill=(255, 255, 255)
            )
        draw.rectangle(
            (1, y[2+products_len]+1, x[-1]-1, y[3+products_len]-1),
            fill=rgb
        )
        draw.rectangle(
            (1, y[3+products_len+channel_len]+1,
             x[-1]-1, y[4+products_len+channel_len]-1),
            fill=rgb
        )
        draw.rectangle(
            (1, y[4+products_len+channel_len+operator_len]+1,
             x[-1]-1, y[5+products_len+channel_len+operator_len]-1),
            fill=rgb
        )
        if int(time_str()) < 10:
            d_str = p['startTime']
        else:
            d_str = p['startTime']+"    " + \
                re.compile(' '+'(\d{1,2}:\d{2})',
                           re.S).findall(p['updateTime'])[0]
        hd = fontbdh.getsize(d_str)
        draw.text(
            (
                int((x[-1]-hd[0])/2),
                3,
            ),
            d_str,
            (0, 0, 0),
            font=fontbdh
        )
        return im1
    else:
        return 0


pt = {
    "dc": {
        "name": "贷超",
        "url": 'http://prod.op.stats.raipeng.cn/market-stats/channelDataOfCreditLoan/loan/list',
        "rgb": (153, 171, 218)
    },
    "xd": {
        "name": "小贷",
        "url": 'http://prod.op.stats.raipeng.cn/credit-stats/channelDataOfCreditLoan/loan/list',
        "rgb": (112, 173, 71)
    },
    "hy": {
        "name": "回Y",
        "url": 'http://prod.op.credit.statis.raipeng.cn/channelDataOfCreditLoan/special/list',
        "rgb": (255, 192, 0)
    }
}


if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    null = ''
    filename1 = "C:/Windows/Fonts/msyh.ttc"
    filename2 = "C:/Windows/Fonts/msyhbd.ttc"

    fontbdh = ImageFont.truetype(filename2, 52, encoding="utf-8")
    fontbd = ImageFont.truetype(filename2, 32, encoding="utf-8")
    font = ImageFont.truetype(filename1, 26, encoding="utf-8")
    if int(time_str()) >= 10:
        d = 0
    else:
        d = 1
    # 从剪贴板加载，格式为灰度
    im_arr = np.array(ImageGrab.grabclipboard().convert('L'))
    #二值化
    im_arr=np.where(im_arr>120,1,0)
    #找出黑色像素的最大坐标和最小坐标
    mi=np.where(im_arr==0)
    a1=np.min(mi[0])
    b1=np.min(mi[1])
    ma=np.where(im_arr==0)
    a2=np.max(ma[0])
    b2=np.max(ma[1])
    #还原为有效的数组大小，转换为1维数组
    im_arr=im_arr[a1+3:a2-2,b1+3:b2-2]
    im_arr=im_arr.reshape((im_arr.shape[0]*im_arr.shape[1],))
    #数组转换为字符串
    im_str=''.join(str(i) for i in im_arr)
    #分割成长度8的字符串列表
    bin_lis=re.findall(r'.{8}', im_str)
    #将二进制字符串转换成utf8编码
    r_str=bytes(int(x,base=2) for x in bin_lis).decode('utf-8')
    r_str=r_str.replace('\0','')
    r_str=eval(r_str)

    r1, j1 = updt(r_str[1], d)
    r2, j2 = updt(r_str[2], d)
    r3, j3 = updt(r_str[3], d)
    ra=[r1,r2,r3]
    ja=[j1,j2,j3]
    pa=["xd","dc","hy"]
    jk,im,sizex,sizey=[],[],[],[]
    if -1 in ja:
        tkinter.messagebox.showinfo("error", "出错啦")
    else:
        for i in range(len(ja)):
            if ja[i]>0:
                jk.append(i)
                imb=table_im(ra[i], pt[pa[i]]["rgb"])
                im.append(imb)
                sizex.append(imb.size[0])
                sizey.append(imb.size[1])

        sizeyt=[0]
        for i in range(len(sizex)):
            sizet=sizey[i]
            if sizex[i]<max(sizex):
                sizet=int(sizet*(max(sizex)/sizex[i]))
                im[i]=im[i].resize((max(sizex),sizet))
            sizeyt.append(sizet)
        x, y = max(sizex), sum(sizeyt)
        ima = Image.new("RGB", (x, y), "#ffffffff")
        print(sizeyt)
        for i in range(len(im)):
            ima.paste(im[i], (0, sum(sizeyt[0:i+1])))
        ima.show()
        
        output= BytesIO()
        ima.convert("RGB").save(output,"BMP")
        data=output.getvalue()[14:]
        output.close()
        output = BytesIO()
        ima.convert("RGB").save(output, "BMP")
        data = output.getvalue()[14:]
        output.close()
        wc.OpenClipboard()
        wc.EmptyClipboard()
        wc.SetClipboardData(win32con.CF_DIB, data)
        wc.CloseClipboard()

