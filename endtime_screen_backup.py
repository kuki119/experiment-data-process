#coding:utf-8
#计算筛分效率2017.3.25__可以使用Excel通过自行排序的方式验证程序运行结果是否正确
#2017.4.26修改，增加输出斜率拐点，输出拐点时刻与该时刻的单位时间筛分效率
import numpy as np
import matplotlib.pyplot as plt
import xlwt

def loadData(address_file):
#导入数据！！
    import csv
    data = []
    with open(address_file,newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        del(rows[0:9])
        for i in rows:
            try:
                if len(i[1]) > 0:
                    a = float(i[1])
                    data.append(a)
            except:
                continue
    return data

def dataSplit(data,num_time):
#按各个时间点，分割数据,包含时间列
    length = len(data)
    length_each_time = length/num_time
    data_time = np.arange(length,dtype=np.float64)#data_time用来存放每个时间点的数据，是个6*8001的矩阵，行首放时刻值
    for i in range(length):
        data_time[i] = data[i]
    # print(data_time.shape,num_time)
    data_time = data_time.reshape(num_time,-1)
    return data_time

def getTimeList(matrix):
#抽出传入矩阵的第一列，成为一个列表
    [row,col] = matrix.shape
    t_list = []
    for i in range(row):
        t_list.append(matrix[i][0])
    # print (t_list)
    return t_list

def dataPlot(x,z):
#画出筛分结果图像，看效果
    plt.scatter(x,z,alpha=0.3,marker='.')
    # plt.axvline(x=x_max,color='red')
    # plt.axhline(y=z_max,color='red')
    # plt.axhline(y=z_min,color='red')
    plt.xticks(np.arange(-80,80+20,5),rotation=60)  ###！！ xticks函数！！设置x轴的刻度，rotation设置刻度标签的旋转！！！
    #xticks()还可以加一个参数，设置x标签的显示，比如各个颗粒形状，以字符串列表的形式放入！！
    # plt.show()

def underScreenFlag(x,z,x_min,x_max,screen_pms):
#通过x，z坐标，判断出筛下颗粒位置矩阵，布尔矩阵
    x_flag = (x>x_min)&(x<x_max)
    z_flag = z<(x*screen_pms[0] + screen_pms[1]) 
    flag = x_flag & z_flag
    # print(flag)
    return flag

def extractData(matrix,flag):
#完成将数值矩阵，通过布尔矩阵，抽出其值为True的值
    [row,col] = matrix.shape
    a = np.ones(row*col,dtype='int');a = a.reshape(row,col)
    a = a&flag
    extract_matrix = a*matrix
    # print(extract_matrix)
    return(extract_matrix)

def delFirstCol(matrix):
#数据导入完成后，第一列存放时刻，用该函数剔除第一列
    [row,col] = matrix.shape
    # a = np.zeros(row).reshape(row,-1)
    # b = np.ones(col)
    # b[0] = 0
    # c = a+b
    # matrix_no_first_col = matrix * c
    # # print(matrix_no_first_col[0][2])

    new_matrix = matrix[:,1:col+1]
    # print (new_matrix.shape,new_matrix[0,7753]) #验证正确
    return(new_matrix)

def apartMassMatrix(mass,diameter):
#计算传入的质量矩阵中，大于标定质量的质量矩阵,只需要返回0-1矩阵即可，便于统计颗粒个数
    new_mass = delFirstCol(mass)
    [row,col] = new_mass.shape
    ones_matrix = np.ones(row*col,dtype='int').reshape(row,-1) 
    density = 2678  #密度2678千克每立方米
    diameter = diameter/1000
    mass_standard = ((4.0/3.0)*np.pi*density*(diameter/2.0)**3)*1000 #把质量的kg单位变成g单位
    large_particles_flag = (new_mass > mass_standard) | (new_mass == mass_standard)
    # large_particles_flag = new_mass > 0
    small_partical_flag = (new_mass < mass_standard) & (new_mass > 0)
    # large_particles_flag = delFirstCol(large_particles_flag)#去除时间列
    # small_partical_flag = delFirstCol(small_partical_flag)
    l_matrix = extractData(ones_matrix,large_particles_flag) #把大于分离粒径的颗粒矩阵抽出
    s_matrix = extractData(ones_matrix,small_partical_flag) #把小于分离粒径的颗粒矩阵抽出
    # l_matrix = extractData(a,large_particles_flag) #把大于分离粒径的颗粒矩阵抽出
    # s_matrix = extractData(a,small_partical_flag) #把小于分离粒径的颗粒矩阵抽出
    # print(mass_standard)
    return l_matrix,s_matrix

def addEachCol(matrix):
#分别把矩阵的每一行加起来，放到一个一维数组里，用于后面的计算
    [row,col] = matrix.shape
    a = np.arange(row)
    for i in range(row):
        a[i] = float(np.sum(matrix[i]))
    # print (a,a.shape)
    return a

def efficiency(large_flag,small_flag,under_large_flag,under_small_flag):
#计算筛分效率
    # [row,col] = large_flag.shape
    large_sum = addEachCol(large_flag)
    small_sum = addEachCol(small_flag)
    under_large_sum = addEachCol(under_large_flag)
    under_small_sum = addEachCol(under_small_flag)

    eff1 = (under_small_sum/small_sum)*100
    eff2 = (under_large_sum/large_sum)*100
    eff = eff1 - eff2
    # print(eff1,'\n',eff2,'\n',eff)
    return eff1,eff2,eff

def outPutData(row,data,end,worksheet):
#导出数据
    head = ['时刻(s)','eff1(%)','eff2(%)','eff(%)','单位时间筛分效率(%)','未筛颗粒个数','所占百分比(%)','斜率']
    # print(data)
    for i in range(len(head)):
        if row == -1:
            worksheet.write(0,i,head[i])
        else:
            worksheet.write(row+1,i,float(data[i]))
            # workbook.save('export.xls')
    # if row == end-1:
        # workbook.save('export1.xls')
    

def upScreen(x,z,mass,x_min,x_max,screen_pms):
#判断筛上颗粒, 仅仅输入points 中的x_min,x_max
    x = delFirstCol(x)
    z = delFirstCol(z)
    mass = delFirstCol(mass)
    # mass_array = addEachCol(mass)  #用于统计质量
    x_flag = (x<x_max) & (x>x_min)
    z_flag = z>(x*screen_pms[0] + screen_pms[1]) 
    flag = x_flag & z_flag
    # mass_up_matrix = extractData(mass,flag)  #用于统计质量
    # mass_up_array = addEachCol(mass_up_matrix)  #用于统计质量
    [row,col] = mass.shape
    ones_matrix = np.ones(row*col,dtype='int').reshape(row,-1)
    num_up_matrix = extractData(ones_matrix,flag)
    num_up_array = addEachCol(num_up_matrix)
    # print(flag)
    return (num_up_array,(num_up_array/float(col))*100)

def calSlope(time,num_list):
    # print(len(time),num_list)
    x1 = np.array(time[:-1],dtype='float')
    x2 = np.array(time[1:],dtype='float')
    num1 = np.array(num_list[:-1])
    num2 = np.array(num_list[1:])
    slope = (num2-num1)/(x2-x1)
    slope_list = list(slope)
    # print(slope_list)
    return (slope_list)

def get4Point():
#从散点图上取得4个点，分别是筛网水平方向的左右极限点，和筛网直线上两点
#挑出x坐标最大与最小的点，作为筛网的水平范围，剩余两点则是筛网直线
    points = plt.ginput(4)
    p = sorted(points)
    return(p)

def calLine(points):
#根据传入的两点坐标，计算通过这两点的直线参数方程
    point1 = np.array(points[0],dtype=float)
    point2 = np.array(points[1],dtype=float)
    vector = point1 - point2
    k = vector[1]/vector[0]
    b = point1[1] - k*point1[0]
    return(k,b)

def myfind(value,my_list):
#在 my_list里找 大于 value 值的第一个
    length = len(my_list)
    for i in range(length):
        if my_list[i] == 0:
            continue
        if my_list[i] > value:
            return(my_list[i],i)

def main():
    #需要修改的地方！！
    ########################
    address = 'F:/self_orthogonal_experiment/data/test41'
    num_time = 10 #表示自己选取的时刻数，可选5个时刻，或7个时刻……可以通过数数据输出时停顿多少下，n-1
    
    ########################

    address_x = address+'/x.csv'
    address_z = address+'/z.csv'
    address_mass = address+'/mass.csv'

    # num_partical = 8000 #打开导出数据文件，按第一组数据的长度为颗粒数量，通常是7998~8000
    # num_time = 10 #表示自己选取的时刻数，可选5个时刻，或7个时刻……

    #对数据进行导入和分块处理
    x = loadData(address_x)
    x_matrix = dataSplit(x,num_time)
    # print(sum(x_matrix[5][1:]))  #用于验证程序导出数据是否正确，可以
    z = loadData(address_z)
    z_matrix = dataSplit(z,num_time)
    # print(sum(z_matrix[5][1:]))  #用于验证程序导出数据是否正确，可以
    mass = loadData(address_mass)
    mass_matrix = dataSplit(mass,num_time)
    time_list = getTimeList(mass_matrix)
    # print(time_list)
    # print(sum(mass_matrix[0]))
    # print(sum(mass_matrix[5][1:]))  #用于验证程序导出数据是否正确，可以
    
    #python出图太慢，可以用matlab出图，然后在python中计算
    dataPlot(x_matrix[0][1:],z_matrix[0][1:])
    # dataPlot(x_matrix[2][1:],z_matrix[2][1:],z_min,z_max,x_max)

    #获取筛上区域 取四个点，分别确定 x_min x_max 和筛网
    points = get4Point()
    params = calLine(points[1:3])
    x_min = points[0][0]
    x_max = points[-1][0]


    #通过X Z矩阵的布尔运算，确定筛下颗粒的位置矩阵
    flag_position = underScreenFlag(x_matrix,z_matrix,x_min,x_max,params)# z_max=-35,z_min=-45,x_max=80,视情况而定
    under_mass_matrix = extractData(mass_matrix,flag_position)
    # print(under_mass_matrix.shape)
    #返回总颗粒中，大于分离粒径的颗粒质量矩阵和小于分离粒径的颗粒质量矩阵的0-1矩阵
    [large_mass_flag,small_mass_flag] = apartMassMatrix(mass_matrix,0.8) #0.8是设定的分离粒径（直径） 单位mm，已使用mass_matrix验证正确
    #返回筛下，大于分离粒径的颗粒质量矩阵和小于分离粒径的颗粒质量矩阵的0-1矩阵
    [under_large_mass_flag,under_small_mass_flag] = apartMassMatrix(under_mass_matrix,0.8) #0.8是设定的分离粒径（直径） 单位mm，已使用mass_matrix验证正确

    #计算筛分效率
    [eff1,eff2,eff] = efficiency(large_mass_flag,small_mass_flag,under_large_mass_flag,under_small_mass_flag)
    # print(eff1.shape,'\n',eff2.shape,'\n',type(eff))

    #统计未筛颗粒的个数，占总个数的百分比，质量，占总质量的百分比
    [up_grain_num,up_grain_percent] = upScreen(x_matrix,z_matrix,mass_matrix,x_min,x_max,params)
    # print(up_grain_num,up_grain_percent)

    time_np = np.array(time_list,dtype='float')
    eff_unit_time = eff/time_np
    slope = calSlope(time_list,up_grain_num) #这里斜率的计算是使用 未筛颗粒个数与时刻
    #因为斜率值比数据值少一位，所以通过两次翻转，在第一位加0，注意np.ndarray不能用reverse(),所以转成list
    slope.reverse()
    slope.append(0)
    slope.reverse()
    print(slope)
    # print(eff1,eff2,eff,eff_unit_time)

    # #计算斜率集合中，前一点是后一点的几倍
    # slope_len = len(slope)
    # slope_div = []
    # for i in range(1,slope_len):
    #     slope_div.append([slope[i-1]/slope[i],i-1])

    # print(u'最大的斜率变化倍数:{}'.format(max(slope_div)))
    # print(u'时间点:{}'.format(time_list[max(slope_div)[1]]))
    # print(u'该时刻的单位时间筛分效率:{}'.format(eff_unit_time[max(slope_div)[1]]))

    #设定斜率为-1000时为筛分结束，找斜率大于-1000的第一个时刻,将该时刻的前一个时刻定为筛分结束时刻：
    [slope_1000,ite] = myfind(-1000,slope)
    end_time = time_list[ite-1]
    eff_end_time = eff_unit_time[ite-1]
    print(u'斜率大于-1000的第一个斜率为：{}'.format(slope_1000))
    print(u'时间点：{}'.format(end_time))
    print(u'该时刻的单位时间筛分效率：{}'.format(eff_end_time))


    # #导出数据 导出时刻、筛分效率(eff1,eff2,eff)、未筛颗粒个数、所占百分比。
    # workbook = xlwt.Workbook()
    # worksheet = workbook.add_sheet('export_data')
    # end = len(time_list)
    # for i in range(end):
    #     data = []
    #     data.append(time_list[i])
    #     data.append(eff1[i])
    #     data.append(eff2[i])
    #     data.append(eff[i])
    #     data.append(eff_unit_time[i])
    #     data.append(up_grain_num[i])
    #     data.append(up_grain_percent[i])
    #     data.append(slope[i])
    #     outPutData(i,data,end,worksheet)
    #     if i == end-1:
    #         outPutData(-1,time_list,end,worksheet)
    # address_export = address+'/export.xls'
    # workbook.save(address_export)
        

if __name__=='__main__':
    main()
    


