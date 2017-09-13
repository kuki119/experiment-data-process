#coding:utf-8
#输入边长和水的体积，返回水的高度---4.16
#直接使用四边的直线方程，并且上下底面为矩形
#使用小梯形台体积叠加的方法 逼近三重积分
import math
import numpy as np
# from sympy import *

def inLenVolume():
#输入水箱的参数  转化为 各个面的三点参数 用于计算各个面方程
#用bot_bot,bot_top,bot_side 分别表示底面下底边、底面上底边、底面腰
#用top_bot,top_top,top_side 分别表示顶面下底边、顶面上底边、顶面腰
    # bot_long = float(input(u'输入底面矩形长cm：'))
    # bot_short = float(input(u'输入底面矩形宽cm：'))
    # top_long = float(input(u'输入顶面矩形长cm：'))
    # top_short = float(input(u'输入顶面矩形宽cm：'))
    # height = float(input(u'输入水箱高度cm：'))
    volume = float(input(u'输入水的体积L：'))*1000

    #测试模型尺寸：
    bot_long = 36.4
    bot_short = 17
    top_long = 32
    top_short = 14
    height = 20

    return (bot_long,bot_short,top_long,top_short,height,volume)

def calCoordinate(bot_long,bot_short,top_long,top_short,height):
#建立空间直角坐标系 将垂直壁与xoz面重合  将底面与xoy面重合 z轴与垂直面中线重合
#A\B\C\D:底面四点，逆时针  E\F\G\H:顶面四点，顺时针
#用A-H \ B-G \ C-F \ D-E 计算直线方程
    A = np.array([bot_long/2,0,0],dtype=float)
    B = np.array([bot_long/2,bot_short,0],dtype=float)
    C = np.array([-bot_long/2,bot_short,0],dtype=float)
    D = np.array([-bot_long/2,0,0],dtype=float)

    E = np.array([-top_long/2,0,height],dtype=float)
    F = np.array([-top_long/2,top_short,height],dtype=float)
    G = np.array([top_long/2,top_short,height],dtype=float)
    H = np.array([top_long/2,0,height],dtype=float)
    # print(A,'\n',B,'\n',C,'\n',D,'\n',E,'\n',F)
    
    return(A,B,C,D,E,F,G,H)

def calLine(point1,point2):
#根据空间两点的坐标，计算空间直线的参数方程
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

def calSurface(l_matrix,z):
#计算指定Z值处的矩形面积

    regu_long = 2*calXY(l_matrix,z)[0]
    regu_short = calXY(l_matrix,z)[1]
    s = regu_short*regu_long

    return(s)

def main():

    #设置循环步长 delta_h
    delta_h = 0.0001
    # delta_h = 1.0
    # delta_h = 10.0

    [bot_long,bot_short,top_long,top_short,height,volume] = inLenVolume()
    [A,B,C,D,E,F,G,H] = calCoordinate(bot_long,bot_short,top_long,top_short,height)

    #计算空间直线参数方程,用 l_1 l_2 l_3 l_4分别表示四条边线 逆时针
    #用A-H \ B-G \ C-F \ D-E 计算四条边线的直线方程
    l_fs_matrix = calLine(B,G)


    # s = calSurface(0)
    # print(s)
    #循环计算Z值
    z = 0; v = 0
    while v < volume:
        z = z + delta_h
        s = calSurface(l_fs_matrix,z)
        v = v + s*delta_h
        
    print(u'当体积为{}mL时，高度为{}'.format(volume,z))

if __name__=='__main__':
    main()