#coding:utf-8
#x轴数据分别代表：振动频率、振动幅度、振动方向角、摆动频率和摆动角度。
#y轴数据代表单位时间筛分效率
#计算不同颗粒形状颗粒 在一个x刻度下的y值的均值，平移曲线，使各点与均值的距离平方和最小
#得到各个曲线所需要的平移距离a，输出平移后的数据列表
#!!注意修改变量名称！！！  columns名称 sheet名称 输出路径名称

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

xlsx = pd.ExcelFile('D:\\try\summary\summary\Average-9.xlsx')

df = pd.read_excel(xlsx,'Sheet1')

#将 DataFrame 数据转成矩阵，第一行为 NAN
data_m = df.as_matrix(columns=['Time','Average'])
t = data_m.T[0][1:]
avg1 = data_m.T[1][1:]

#采样频率
fs = len(t)
print(fs)

#时间序列的长度
len_t = len(t)
n = np.arange(len_t).reshape(1,-1)

#数字角频率的取值范围
w = np.pi*np.arange(0,10,0.05).reshape(1,-1)
f = w*fs/(2*np.pi)
nf = np.dot(n.T,f)
exp_nf = np.exp(-1j*nf) 

#不知道a1是行向量还是列向量
avg2 = np.dot(avg1,exp_nf)

#取绝对值 取模
abs_avg2 = abs(avg2)

#将两个向量组成矩阵
matrix = np.vstack((f,abs_avg2))

#将矩阵转成 DataFrame
data_out = pd.DataFrame(matrix.T,columns=['w','average']) 
print(data_out)

#数据输出
data_out.to_excel('D:\\try\summary\summary\\average_9_out.xlsx',sheet_name='sheet1')