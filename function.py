#coding:utf-8
import matplotlib.pyplot as plt 
import numpy as np

def calLine(points):
#根据传入的两点坐标，计算通过这两点的直线参数方程
    point1 = np.array(points[0],dtype=float)
    point2 = np.array(points[1],dtype=float)
    vector = point1 - point2
    k = vector[1]/vector[0]
    b = point1[1] - k*point1[0]

    return(k,b)

x = np.arange(1,10,0.1)
y = 3*x
plt.plot(x,y)
pm = calLine(plt.ginput(2))
print(pm)