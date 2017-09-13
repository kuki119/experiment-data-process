# coding:utf-8

import numpy as np 
from scipy.optimize import leastsq
import pylab as pl 

def func(x,p): #编写数据拟合所用的函数 A*sin(2*pi*k*x + theta)
    A, k, theta = p
    return A*np.sin(2*np.pi*k*x+theta)

def residuals(p,y,x): #实验数据x,y和拟合函数之间的差，p为拟合需要找的系数
    return y-func(x,p)

x = np.linspace(0,-2*np.pi,100)
A,k,theta = 10,0.34,np.pi/6 #真实的实验参数
y0 = func(x,[A,k,theta]) #用于产生理想的实验数据，对比值
y1 = y0 + 2*np.random.randn(len(x)) #在理想数值上产生噪声

p0 = [100,0.4,0] #初试化的参数值

plsq = leastsq(residuals,p0,args=(y1,x))#求最小二乘的函数，传入误差值，初试参数和实验数据列表

print u"真实参数：",[A,k,theta]
print u"拟合参数",plsq[0]


pl.plot(x,y1,label=u"the number with error")
pl.plot(x,func(x,plsq[0]),label=u"fitting data")
pl.plot(x,y0,label=u"the idol number")
pl.legend()  ##显示标签
pl.show()