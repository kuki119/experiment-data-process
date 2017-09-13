#coding:utf-8
#可以通过ginput命令 在图像上取点，直接用于后续计算！！！
#用于寻找筛面的直线方程！！！
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

# x = np.arange(1,10,0.1)
# y = 3*x
# plt.plot(x,y)
# points = plt.ginput(2)
# print(type(points[0]))
# pm = calLine(points)
# print(pm)


# #写一个在四个点里找出横坐标最大的和最小的两个点
# p1=(random.random(),random.random())
# p2=(random.random(),random.random())
# p3=(random.random(),random.random())
# p4=(random.random(),random.random())
# points=(p1,p2,p3,p4)

# p_min = min(points)
# p_max = max(points)

# print('points:{}'.format(points))
# print('p_min:{}'.format(p_min))
# print('p_max:{}'.format(p_max))

#写一个以data_id 为key值，x y z mass为value的字典函数
def makeDict(data):
    data_dict = {}
    for i in range(len(data.x)):
        data_dict[i] = [data.x[i],data.y[i],data.z[i],data.mass[i]]

    print(len(data_dict))
    return(data_dict)

def myfind(value,my_list):
#在 my_list里找, 之后数值都是降序排列，且大于-1000的第一个数值
    length = len(my_list)
    for i in range(1,length):
        if my_list[i] > value:
            j = i+1
            #判断之后是不是升序排列
            while j<length:
                print(j,my_list[j])
                if my_list[j-1] < my_list[j] :
                    j += 1
                    continue
                else:
                    break 
            if j == length:
                return(my_list[i],i)



a = [0,12,24,23,26,34,12,45,69,78,13,26,33,42,23,28,39,40,48]
[value,ite] = myfind(20,a)

print(value,ite)

def calIter(time_start,time_end):
#根据外部输入的入料结束时间，仿真结束时间，来计算需要的间隔iterate

    ite = (time_end-time_start)*100//10

    return(ite)


b = [['TIME:', '0.4'],['Q01 : Particle Position X:'],['', '-72.5767'],
 ['', '-74.4076'],['', '-72.4922'],['', '-70.3896'],['', '-75.3787'],
 ['', '-74.6942'],['', '-74.9247'],['', '-72.6878']]
b = np.array(b,dtype='float')
c = []
for i in b.ravel():
    try:
        print(i)
        e = float(i)
        c.append(e)
    except:
        continue
