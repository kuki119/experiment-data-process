# coding:utf-8
##插值计算
import numpy as np
import pylab as pl
from scipy import interpolate

x = np.linspace(0,2*np.pi+np.pi/4,10)
y = np.sin(x)

x_new = np.linspace(0,2*np.pi+np.pi/4,100)
f_linear = interpolate.interp1d(x,y)   
#线性插值 此处的x,y用于传入计算插入直线的方程，一小段一小段直线方程，后由x_new计算插入的函数值
tck = interpolate.splrep(x,y)
#用splrep计算样条曲线的参数，然后将参数传递给splev函数，计算各个取样点的插值结果
y_bspline = interpolate.splev(x_new,tck) #B-spline 样条插值（spline样条，interpolate插值）

pl.plot(x,y,'o',label='initial number')
pl.plot(x_new,f_linear(x_new),'b+',label='linear interpolate')
pl.plot(x_new,y_bspline,'r.',label='B-spline interpolate')
pl.legend()
pl.show()