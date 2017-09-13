#coding:utf-8
#x轴数据分别代表：振动频率、振动幅度、振动方向角、摆动频率和摆动角度。
#y轴数据代表单位时间筛分效率
#计算不同颗粒形状颗粒 在一个x刻度下的y值的均值，平移曲线，使各点与均值的距离平方和最小
#得到各个曲线所需要的平移距离a，输出平移后的数据列表

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

xlsx = pd.ExcelFile('D:\document\yue\data_in.xlsx')
sheet_list = ['vbfrequency','vbamplitude','vbstroke','wavefrequency','waveangle']
#取出各个sheet中的数据
vbf = pd.read_excel(xlsx,'vbfrequency')
vbam = pd.read_excel(xlsx,'vbamplitude')
vbstr = pd.read_excel(xlsx,'vbstroke')
wvf = pd.read_excel(xlsx,'wavefrequency')
wvang = pd.read_excel(xlsx,'waveangle')

with pd.ExcelWriter('D:\document\yue\data_out.xlsx') as writer:

    for k,df in enumerate([vbf,vbam,vbstr,wvf,wvang]):
        x = df.columns[1:]
        averages = []
        rows = len(df.index)
        cols = len(x) #不算第一列的颗粒形状列，用于作后面的分母

        #计算每一列的均值
        for i in x:
            averages.append(np.average(df[i]))
        print('averages:',averages)

        #做一个均值的矩阵，用于后面的直接求差
        avg_m_T = np.ones((cols,rows))
        for i in range(cols):
            avg_m_T[i] = avg_m_T[i] * averages[i]
        avg_m = avg_m_T.T
        print(avg_m)

        #将原 dataframe 转成矩阵
        y_m_T = np.arange(rows*cols,dtype='float').reshape(cols,rows)
        for i,j in enumerate(x):
            y_m_T[i] = df[j]
        y_m = y_m_T.T
        print(y_m)

        # #将 DataFrame 转成矩阵的简单做法
        # df.as_matrix(columns=None)
        # #df.as_matrix(columns=[18,23,28,28]) #即可以指定转换第几列

        #画出平移前的数据点图
        fig,(ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(15,7))
        for i in range(rows):
            ax1.plot(x,y_m[i])
        ax1.set_title('moved before')
        # plt.show()

        #拿两个矩阵相减
        ans_m = y_m_T - avg_m_T
        a = -sum(ans_m)/cols
        print(a)

        #计算移动后的数据点
        y_m_after = y_m
        for i in range(rows):
            y_m_after[i] = y_m[i] + a[i]
        print('moved after:','\n',y_m_after)

        #画出移动后的图像
        for i in range(rows):
            ax2.plot(x,y_m_after[i])
        ax2.set_title('moved after')
        ax2.legend('abcdefghis')
        plt.show()

        #将已得数据转成DataFrame:
        data_out = pd.DataFrame(y_m_after,columns=list(x))
        data_out['A'] = a

        #数据输出
        data_out.to_excel(writer,sheet_name=sheet_list[k])

