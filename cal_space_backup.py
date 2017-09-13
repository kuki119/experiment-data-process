#coding:utf-8
#计算松散系数：筛面上料层中颗粒之间三维距离 / 料层中颗粒的平均粒径  --4/22
#整个编程想法：
#1、先从数据库中将全部的数据取出；2、画出散点图，用尺子选择直线上的点，从而确定筛网直线；
#3、用数据库语句将筛网上的数据全部取出；4、计算筛上颗粒的三维距离；5、计算筛上颗粒平均粒径
#遗留问题：
#1、尝试在颗粒散点图上画筛网直线，失败；2、将数据库导出的数据精度降低，提高运算速度
#3、看后期运算速度，如果很慢就将字典改矩阵运算；4、在看松散系数定义，颗粒距离总值较大，与颗粒粒径均值之比很大

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import pymysql
from math import pi

def loadTotalData():
    #将数据库中数据导入！用data.x[1]的形式调用
    dbconn = pymysql.connect( host='localhost',user='root',password='1029384756',database='xyz_mass')
    sqlcmd='SELECT * FROM time_08;'
    data_total = pd.read_sql(sqlcmd,dbconn)
    return (data_total)

def loadUpData(line_pm,p_min,p_max):
    #用数据库，将筛网以上区域内的颗粒数据取出 
    x_min = str(p_min[0]); x_max = str(p_max[0]) 
    k = str(line_pm[0]); b = str(line_pm[1])
    dbconn = pymysql.connect( host='localhost',user='root',password='1029384756',database='xyz_mass')
    sqlcmd='SELECT x,y,z,mass FROM time_08 WHERE z > x*'+k+'+'+b+' AND x >'+x_min+' AND x<'+x_max+';'
    data = pd.read_sql(sqlcmd,dbconn)
    print('the num from database:{}'.format(len(data.x)))
    return (data)

def drawScatter(x,y,c):
    #画个散点图看看,并且从中取出大部分筛上颗粒的 大概区域
    plt.scatter(x,y,color=c,marker='.',alpha=0.3)
    # points = plt.ginput(2)
    # return(points)

def drawLine(x_min,x_max,pm):
#不知道为什么，不能在原图的基础上画直线！！问题！！！
    k = pm[0]
    b = pm[1]
    x = np.arange(x_min,x_max,1)
    y = k*x + b
    plt.plot(x,y,'g-')
    print(len(x),len(y))

def calLine(points):
#根据传入的两点坐标，计算通过这两点的直线参数方程
    point1 = np.array(points[0],dtype=float)
    point2 = np.array(points[1],dtype=float)
    vector = point1 - point2
    k = vector[1]/vector[0]
    b = point1[1] - k*point1[0]
    return(k,b)

def get4Point():
#从散点图上取得4个点，分别是筛网水平方向的左右极限点，和筛网直线上两点
#挑出x坐标最大与最小的点，作为筛网的水平范围，剩余两点则是筛网直线
    points = plt.ginput(4)
    p = sorted(points)
    return(p)

def makeDict(data):
#写一个以data_id 为key值，x y z mass为value的字典函数，且values值为np.array
    data_dict = {}
    for i in range(len(data.x)):
        data_dict[i] = np.array([data.x[i],data.y[i],data.z[i],data.mass[i]],dtype=np.float32)

    print('the num from dictionary:{}'.format(len(data_dict)))#该值与上面loadUpData的输出数据一样则正确
    print('the pirticle\'s x/y/z/mass above screen:{}'.format(data_dict[1]))
    return(data_dict)

def calAveDemeter(data_db):
#将筛上颗粒的字典传入，用质量 密度 直径之间的关系，求出全部颗粒的直径和 并求平均值
    density = 2678.0/1000000.0  #密度2678千克每立方米 除以10^6转化为克每立方毫米
    volume = data_db.mass[:]/density
    demeter = ((3*volume/(4*pi))**(1.0/3.0))*2
    demeter_ave = sum(demeter)/len(demeter)
    # a = sorted(demeter); print(a[-5:],a[0:5])#可以看到筛上颗粒粒径排序中的前五个和后五个
    # print(demeter_ave)
    return(demeter_ave)

def calDistance(data,num):
#传入一个字典，计算该字典的第一个点，与其他点之间的距离和; num代表现在该第几个颗粒了
#除了用字典计算，还可以尝试用矩阵来计算，每一行都减去第一行，然后与转置矩阵相乘，横向求和 开方 纵向求和
    dis_total = 0
    for i in data.values():
        dis1 = data[num] - i #循环计算第一个点与其他点坐标的差值
        dis2 = (sum(dis1**2))**(0.5)  #计算坐标差值的平方和，之后开方
        dis_total += dis2
    # print(dis_total)
    return(dis_total)

def main():
    fig = plt.figure()
    data = loadTotalData()
    drawScatter(data.x[:],data.z[:],'red')

    #从散点图中取出四个点，用于计算筛网直线和筛网水平范围
    #这四个点中，筛网直线的两点！！需用直尺比着取！！，否则不能保证计算得到的直线会不会穿越筛网!!!!
    [p1,p2,p3,p4] = get4Point()
    pm = calLine((p2,p3))   #pm 为直线的参数
    # print('k={0}, b={1}'.format(pm[0],pm[1]));print('x_min:{}'.format(p1[0]));print('x_max:{}'.format(p4[0]))

    #将计算得到的直线参数用于数据库的数据提取
    # pm=[-0.1657,0.1956];p1=(-78.239,10);p4=(77.7896,23)
    data_up = loadUpData(pm,p1,p4)
    # drawScatter(data_up.x[:],data_up.z[:],'black')

    #将筛上的颗粒数据编辑成字典的形式，转成字典是为了计算距离方便，计算质量还用data_up
    data_dict = makeDict(data_up)

    #计算筛上颗粒的平均粒径，注意data_up是来自于数据库
    demeter_ave = calAveDemeter(data_up)

    #循环计算每个颗粒与其他颗粒距离之和，注意 计算过的点删去，即无重复距离段
    # main函数中的dis_total指的是所有的距离之和，而calDistance函数中的dis_total仅指一个颗粒与其他颗粒的距离之和
    dis_total = 0
    length_data_dict = len(data_dict)
    for i in range(length_data_dict):
        dis_one = calDistance(data_dict,i)
        dis_total += dis_one
        del data_dict[i]
    print('the total distance between the pirticle:(mm)  {}'.format(dis_total))
    dis_average = dis_total/length_data_dict
    print('the average distance between the pirticle:(mm){}'.format(dis_average))


    #不知道为什么不能在原散点图的基础上画直线，待解决
    # x_min = np.min(data.x); x_max = np.max(data.x)
    # drawLine(x_min,x_max,pm)
    # plt.show()



if __name__=='__main__':
    main()