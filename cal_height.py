#coding:utf-8
#输入边长和水的体积，返回水的高度---4.12难点积分怎么算
import math
import numpy as np
from sympy import *
import matplotlib.pyplot as plt

def inLenVolume():
#输入水箱的参数  转化为 各个面的三点参数 用于计算各个面方程
#用bot_bot,bot_top,bot_side 分别表示底面下底边、底面上底边、底面腰
#用top_bot,top_top,top_side 分别表示顶面下底边、顶面上底边、顶面腰
    # bot_top = float(input(u'输入底面梯形短边长：'))
    # bot_bot = float(input(u'输入底面梯形长边长：'))
    # bot_side = float(input(u'输入底面梯形斜边长：'))
    # top_top = float(input(u'输入底面梯形短边长：'))
    # top_bot = float(input(u'输入底面梯形长边长：'))
    # top_side = float(input(u'输入底面梯形斜边长：'))
    # height = float(input(u'输入水箱高度：'))
    # volume = float(input(u'输入水的体积：'))  
    bot_top = 6.0
    bot_bot = 10.0
    bot_side = 3.61
    top_top = 5.32
    top_bot = 8.0
    top_side = 2.41
    height = 8.0
    volume = 100.0

    return (bot_top,bot_bot,bot_side,top_top,top_bot,top_side,height,volume)

def vecMultiply(vector1,vector2):
#只适用于3维向量叉乘
    [x1,y1,z1] = vector1
    [x2,y2,z2] = vector2
    vector3 = [y1*z2-y2*z1,-x1*z2+x2*z1,x1*y2-x2*y1]
    return(vector3)

def calSurface(point1,point2,point3):
    vector1 = point1 - point2
    vector2 = point3 - point2
    vector_n = vecMultiply(vector1,vector2) #vector_n 即为该平面的法向量
    # vector_n = vecMultiply([1,2,3],[4,5,6]) #验证向量叉乘是否正确
    # print(vector_n)
    vector_n = vecMultiply(vector1,vector2)
    [pm1,pm2,pm3] = vector_n
    [x0,y0,z0] = point1
    # print (pm1,pm2,pm3,-(pm1*x0+pm2*y0+pm3*z0))
    return ([pm1,pm2,pm3,-(pm1*x0+pm2*y0+pm3*z0)])

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

def calLine(vector1,vector2,point):
#计算两平面交线的 空间参数方程  两个向量分别表示两平面的法向量，point表示直线上的点
    vector1 = np.array(vector1,dtype=float)
    vector2 = np.array(vector2,dtype=float)
    vector_k = vecMultiply(vector1,vector2)
    point = np.array(point)
    # print('point:{}'.format(point))
    # print('vector_n:{}'.format(vector_k))
    intercept = point%vector_k
    # print(type(intercept))
    return(vector_k,intercept)

def calInteger(parameter1,parameter2):
#因为该空间梯形台关于yoz面对称，所以只计算一半的体积
    x,y,z = symbols('x,y,z')
    [A1,B1,C1,D1] = parameter1
    [A2,B2,C2,D2] = parameter2
    surface_front_func = A1*x+B1*y+C1*z+D1
    surface_side_func = A2*x+B2*y+C2*z+D2

    step1 = integrate(surface_front_func,(x,0,2))
    print(step1)
    step2 = integrate(step1,( ))
    # print(parameter1,'\n',parameter2)
    # print(surface_front_func,'\n',surface_side_func)

# def drawPlane(parameter1):
#     from mpl_toolkits.mplot3d import Axes3D
#     figure = plt.figure()
#     ax = Axes3D(figure)
#     [A1,B1,C1,D1] = parameter1
#     # [A2,B2,C2,D2] = parameter2
#     xx = np.linspace(-20,20,50)
#     yy = np.linspace(-20,20,50)
#     zz = np.linspace(-20,20,50)
#     XX,YY = np.meshgrid(xx,yy)
#     # ax.plot_surface(XX,YY,0,color='red',alpha=0.3) #xoy面
#     # ax.plot_surface(XX,YY,8,color='red',alpha=0.3) #xoy面
#     # #画x,y,z轴
#     # ax.bar3d(0,0,0,0.1,0.1,20,color='black',alpha=0.3)
#     # ax.bar3d(0,0,0,0.1,20,0.1,color='black',alpha=0.3)
#     # ax.bar3d(0,0,0,20,0.1,0.1,color='black',alpha=0.3)
#     XXY,ZZY = np.meshgrid(xx,zz)
#     ax.plot_surface(XXY,0,ZZY,color='black',alpha=0.3) #xoz面
#     YYX,ZZX = np.meshgrid(yy,zz)
#     ax.plot_surface(0,YYX,ZZX,color='black',alpha=0.3) #yoz面
#     ZZ1 = (A1*xx+B1*yy+D1)/(-C1)
#     ax.plot_surface(XX,YY,ZZ1,color='red',alpha=0.7)
#     # ax.plot_surface(XX,YY,ZZ2,color='black',alpha=0.7)

#     # ax.plot_surface(XX,YY,ZZ2)
#     ax.set_frame_on(True)
#     ax.set_xlabel('axis_x')
#     ax.set_ylabel('axis_y')
#     ax.set_zlabel('axis_z')
#     # plt.show()

def proPlane(surface_pm,point):
#验证这个point 是否在 由该法向量定义的平面上
#仅计算平面误差大小，后续计算默认使用其中三个点坐标所得平面计算
    [x,y,z] = point
    [A,B,C,D] = surface_pm
    err = abs(z-(A*x+B*y+D)/(-C))
    if err > 0.1:
        print(u'所验证的面存在平面度误差！！')
        print(u'平面度误差有：{}'.format(err))
        print(u'修正不修正看你，结果会存在一定误差')

def proLine(matrix_line,point):
#验证选中的point是否在所求的直线上
#matrix_line中 第一行放空间直线参数方程的3个斜率 第二行放空间直线参数方程的3个截距
    [kx,ky,kz] = matrix_line[0]
    [dx,dy,dz] = matrix_line[1]
    [x,y,z] = point
    t = (z-dz)/kz
    err_y = y-ky*t+dy
    err_z = z-kz*t+dz
    print([kx,ky,kz])
    print([dx,dy,dz])
    print([x,y,z])
    print(t)
    print(err_y,err_z)

def main():


    [bot_top,bot_bot,bot_side,top_top,top_bot,top_side,height,volume] = inLenVolume()
    [A,B,C,D,E,F] = calCoordinate(bot_top,bot_bot,bot_side,top_top,top_bot,top_side,height)
    # print (A,'\n',B,'\n',C,'\n',D,'\n',E)
    # test_p1 = np.array([2,-1,4]);test_p2 = np.array([-1,3,-2]);test_p3 = np.array([0,2,3])
    # surface_test = calSurface(test_p1,test_p2,test_p3)
    
    #计算正面和侧面的平面方程
    surface_side = calSurface(A,B,C)
    surface_front = calSurface(C,D,E)
    #验证四个点在不在一个平面上！！！
    proPlane(surface_side,F);proPlane(surface_front,F)
    
    #计算两平面的交线方程
    l_sideback = calLine(surface_side[:3],[0,1,0],B)
    l_frontside = calLine(surface_front[:3],surface_side[:3],C)
    #验证某一个点在不在所求得的直线上
    proLine(l_sideback,A)



    # print(type(surface_front))
    # drawPlane(surface_test)
    # calLine(surface_side[:-1],surface_front[:3],)



if __name__=='__main__':
    main()
