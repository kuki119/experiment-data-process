#coding:utf-8
#输入边长和水的体积，返回水的高度---4.13
#使用小梯形台体积叠加的方法 逼近三重积分
import math
import numpy as np
# from sympy import *

def inLenVolume():
#输入水箱的参数  转化为 各个面的三点参数 用于计算各个面方程
#用bot_bot,bot_top,bot_side 分别表示底面下底边、底面上底边、底面腰
#用top_bot,top_top,top_side 分别表示顶面下底边、顶面上底边、顶面腰
    bot_long = float(input(u'输入底面矩形长cm：'))
    bot_short = float(input(u'输入底面矩形宽cm：'))
    top_long = float(input(u'输入顶面矩形长cm：'))
    top_short = float(input(u'输入顶面矩形宽cm：'))
    height = float(input(u'输入水箱高度cm：'))
    volume = float(input(u'输入水的体积mL：')) 

    bot_top = bot_long
    bot_bot = bot_long
    bot_side = bot_short
    top_top = top_long
    top_bot = top_long
    top_side = top_short

    # bot_top = 36.4
    # bot_bot = 36.4
    # bot_side = 17
    # top_top = 32
    # top_bot = 32
    # top_side = 14
    # height = 20
    # volume = 6000

    return (bot_top,bot_bot,bot_side,top_top,top_bot,top_side,height,volume)

def calCoordinate(bot_top,bot_bot,bot_side,top_top,top_bot,top_side,height):
#建立空间直角坐标系 将垂直壁与xoz面重合  将底面与xoy面重合 z轴与垂直面中线重合
#A:上底面下底边顶点  E:上底面上底边中点  B:下底面下底边顶点  C:下底面上底边顶点  D：下底面上底边中点
#用A B C 计算两腰组成的斜面，用C D E 计算上下两短边组成的斜面
    h_bot = math.sqrt(bot_side**2 - ((bot_bot-bot_top)/2)**2) 
    h_top = math.sqrt(top_side**2 - ((top_bot-top_top)/2)**2)
    A = np.array([top_bot/2,0,height],dtype=float)
    B = np.array([bot_bot/2,0,0],dtype=float)
    C = np.array([bot_top/2,h_bot,0],dtype=float)
    D = np.array([0,h_bot,0],dtype=float)
    E = np.array([0,h_top,height],dtype=float)
    F = np.array([top_top/2,h_top,height],dtype=float)
    # print(A,'\n',B,'\n',C,'\n',D,'\n',E,'\n',F)
    return(A,B,C,D,E,F)

def calLine(point1,point2):
    point1 = np.array(point1,dtype=float)
    point2 = np.array(point2,dtype=float)
    vector = point1-point2
    intercept = point2
    # t = (point2 - intercept)/vector
    # print('calLine{}{}'.format(point1,point2))
    # print('calLine{}{}'.format(vector,intercept))
    # print('calLine{}'.format(t))
    return(vector,intercept)

def calXY(l_matrix,z):
#计算出入矩阵所确定直线的三个坐标
    [kx,ky,kz] = l_matrix[0]
    [dx,dy,dz] = l_matrix[1]
    t = (z-dz)/kz
    x = kx*t + dx
    y = ky*t + dy
    # print('calXY{}'.format([kx,ky,kz]))
    # print('calXY{}'.format([dx,dy,dz]))
    # print('calXY{}'.format([x,y,t]))
    return(x,y)

def calSurface(z):
    global l_bs_matrix,l_fs_matrix
    bot_long = calXY(l_bs_matrix,z)[0]
    bot_short,h = calXY(l_fs_matrix,z)
    s = (bot_long+bot_short)*h/2
    return(s)

def main():

    #设置循环步长 delta_h
    delta_h = 0.01

    [bot_top,bot_bot,bot_side,top_top,top_bot,top_side,height,volume] = inLenVolume()
    [A,B,C,D,E,F] = calCoordinate(bot_top,bot_bot,bot_side,top_top,top_bot,top_side,height)

    #计算空间直线参数方程
    global l_bs_matrix,l_fs_matrix
    l_bs_matrix = calLine(A,B)
    l_fs_matrix = calLine(F,C)

    # s = calSurface(0)
    # print(s)
    #循环计算Z值
    z = 0; v = 0
    while v < volume/2:
        s = calSurface(z)
        v = v + s*delta_h
        z = z + delta_h
    print(u'当体积为{}时，高度为{}'.format(volume,z))

if __name__=='__main__':
    main()