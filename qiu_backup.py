#coding:utf-8
#用18个球填充球形颗粒
#输入被填充的颗粒直径 d_o （1,0.5）
#填充用的小颗粒选用直径 d_i = 0.5 * d_o
#根据空间三个平面，计算填充小球的球心坐标

import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

r_out = np.array([0.25,0.5],dtype=float)
r_inside = 0.5 * r_out #选用外部大球半径的一半的小球来填充

#单独的计算两个r_inside
r = r_inside[0]

#计算 XOY 8个点的坐标
point_1 = np.array([r,0,0])
point_2 = np.array([r*np.sin(np.pi/4),r*np.sin(np.pi/4),0])
point_3 = np.array([0,r,0])
point_4 = np.array([-r*np.sin(np.pi/4),r*np.sin(np.pi/4),0])
point_5 = np.array([-r,0,0])
point_6 = np.array([-r*np.sin(np.pi/4),-r*np.sin(np.pi/4),0])
point_7 = np.array([0,-r,0])
point_8 = np.array([r*np.sin(np.pi/4),-r*np.sin(np.pi/4),0])

#计算 XOZ 6个点的坐标
point_9 = np.array([r*np.sin(np.pi/4),0,r*np.sin(np.pi/4)])
point_10 = np.array([0,0,r])
point_11 = np.array([-r*np.sin(np.pi/4),0,r*np.sin(np.pi/4)])
point_12 = np.array([-r*np.sin(np.pi/4),0,-r*np.sin(np.pi/4)])
point_13 = np.array([0,0,-r])
point_14 = np.array([r*np.sin(np.pi/4),0,-r*np.sin(np.pi/4)])

#计算 YOZ 4个点的坐标
point_15 = np.array([0,r*np.sin(np.pi/4),r*np.sin(np.pi/4)])
point_16 = np.array([0,-r*np.sin(np.pi/4),r*np.sin(np.pi/4)])
point_17 = np.array([0,-r*np.sin(np.pi/4),-r*np.sin(np.pi/4)])
point_18 = np.array([0,r*np.sin(np.pi/4),-r*np.sin(np.pi/4)])

#将散点合并为矩阵
point = np.vstack((point_1,point_2,point_3,point_4,point_5,point_6,
	point_7,point_8,point_9,point_10,point_11,point_12,point_13,point_14,
	point_15,point_16,point_17,point_18))
print(type(point))

#画图散点分布
fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(point.T[0],point.T[1],point.T[2],s=15000)
plt.show()