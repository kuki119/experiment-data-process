# coding:utf-8
#用来计算一重积分、二重积分和三重积分
import numpy as np 
from scipy import integrate

def half_circle(x):
    return (1-x**2)**0.5

N = 10000
x = np.linspace(-1,1,N)
dx = 2.0/N
y = half_circle(x)

#使用integrate的quad函数进行一重积分运算，dblquad函数进行二重积分运算，tplquad函数进行三重积分运算
pi_half,err1 = integrate.quad(half_circle,-1,1)
print pi_half*2

#dblquad函数进行二重积分运算
def half_sphere(x,y):
    return (1-x**2-y**2)**0.5

sphere_half,err2 = integrate.dblquad(half_sphere,-1,1,lambda x:-half_circle(x),lambda x:half_circle(x))

print sphere_half