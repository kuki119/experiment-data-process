#coding:utf-8
#解一个3元的非线性方程组：5*x1+3=0;  4*x0*x0-2*sin(x1*x2)=0;   x1*x2-1.5=0
#解不同的方程组，只需把未知数全部放在等号左边，然后写入f函数的return语句中，
#雅可比矩阵元素为一个方程对各个未知数的求偏导，可加快运算速度，简短时间。
from scipy.optimize import fsolve
from math import sin, cos

def f(x):
    x0 = float(x[0])
    x1 = float(x[1])
    x2 = float(x[2])
    return [5*x1+3, 4*x0*x0 - 2*sin(x1*x2),x1*x2 - 1.5]

def j(x):
    x0 = float(x[0])
    x1 = float(x[1])
    x2 = float(x[2])
    return [[0,5,0],[8*x0,-2*x2*cos(x1*x2),-2*x1*cos(x1*x2)],[0,x2,x1]]

#result = fsolve(f,[1,1,1])#猜测[1,1,1]用来初始化x数组的数值，经过循环验证后逼近准确值(不使用雅可比矩阵)
result = fsolve(f,[1,1,1],fprime=j)#使用雅可比矩阵

print result
print f(result)