#coding：utf-8
#2017.5.7计算松散系数
#计算松散、分层 在入料阶段：可以尝试选择入料口右侧 出料口左侧的部分筛网位置
#或者可以使用入料结束后的阶段，从入料结束到筛分结束选取10个点左右，计算松散和分层
#计算80%的颗粒料层！递增截距params[1],来平移筛网，统计颗粒数目---5.18完成找80%料层
#可以使用筛网的极限点坐标，atan((y2-y1)/(x2-x1)) 算出角度，(b2-b1)*cos(a) 即是80%料层厚度，用于计算体积 ---5.11

import numpy as np
import matplotlib.pyplot as plt
import csv

def loadData(address_file):
#导入数据！！
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

def calLine(x1,y1,x2,y2):
#根据传入的两点坐标，计算通过这两点的直线参数方程
    point1 = np.array([x1,y1],dtype=float)
    point2 = np.array([x2,y2],dtype=float)
    vector = point1 - point2
    k = vector[1]/vector[0]
    b = point1[1] - k*point1[0]
    return k,b

def drawPlot(px,pz,sx0,sx1,sz0,sz1):
    fig,axes = plt.subplots(2,2,figsize=(10,10))
    print(sx0,'\n',sz1,'\n',sx1,'\n',sz0)
    for i,ax in enumerate(axes.ravel()):
        line_p = calLine(sx0[i],sz1[i],sx1[i],sz0[i])
        print(sx0[i],sz1[i],'\n',sx1[i],sz0[i])
        line_x = np.arange(sx0[i],sx1[i])
        line_y = line_x*line_p[0]+line_p[1]
        ax.plot(line_x,line_y,'r')
        ax.scatter(px[i][1:],pz[i][1:],alpha=0.3,marker='.')
        ax.set_title('time={}s'.format(px[i][0]))
    plt.show()

def flagLeft(px_m,sx1):
    #将左侧筛箱的颗粒标记出来
    rows,cols = px_m.shape
    flag = np.ones(rows*cols).reshape(rows,cols)
    for i in range(rows):
        flag[i] = flag[i]*(px_m[i]<sx1[i]) #通过全1矩阵与布尔矩阵相乘，构造满足条件的0-1矩阵
    print('shape of flag_left:',flag.shape)
    return flag

def flagUnder(px_m,pz_m,sx0,sx1,sz0,sz1,z_min):
    #将筛下颗粒标记出来
    try:
        rows,cols = px_m.shape
    except:
        rows = 1; cols = len(px_m)

    flag = np.ones(rows*cols).reshape(rows,cols)
    for i in range(rows):
        try:
            params = calLine(sx0[i],sz1[i],sx1[i],sz0[i])
            bool_m = (px_m[i] < sx1[i]) & (pz_m[i] < px_m[i] * params[0] + params[1]) #构建条件布尔矩阵
            flag[i] = flag[i]*bool_m #通过全1矩阵与布尔矩阵相乘，构造满足条件的0-1矩阵
        except:
            params = calLine(sx0,sz1,sx1,sz0)
            bool_m = (px_m<sx1) & (pz_m<px_m*params[0]+params[1]) #构建条件布尔矩阵
            flag = flag*bool_m
    # print('shape of flag_under:',flag.shape)
    return flag

def flagAbove(px_m,pz_m,sx0,sx1,sz0,sz1):
    #将筛上颗粒标记出来
    try:
        rows,cols = px_m.shape
    except:
        rows = 1; cols = len(px_m)

    flag = np.ones(rows*cols).reshape(rows,cols)
    for i in range(rows):
        try:
            params = calLine(sx0[i],sz1[i],sx1[i],sz0[i])
            bool_m = (px_m[i]<sx1[i]) & (pz_m[i]>px_m[i]*params[0]+params[1]) #构建条件布尔矩阵
            flag[i] = flag[i]*bool_m #通过全1矩阵与布尔矩阵相乘，构造满足条件的0-1矩阵
        except:
            params = calLine(sx0,sz1,sx1,sz0)
            bool_m = (px_m<sx1) & (pz_m>px_m*params[0]+params[1]) #构建条件布尔矩阵
            flag = flag*bool_m
    # print('shape of flag_above:',flag.shape)
    return flag

def numParticles(matrix):
    rows,cols = matrix.shape
    num = []
    for i in range(rows):
        num.append(sum(matrix[i]))
    return np.array(num)

def moveScreen(px_single,pz_single,sx0_single,sx1_single,sz0_single,sz1_single):
#单个时刻计算，每次仅传入一个时刻的数据，在外层进行时刻循环
    flag_above = flagAbove(px_single,pz_single,sx0_single,sx1_single,sz0_single,sz1_single)
    total_num = sum(flag_above.ravel())
    # print('total particles:',total_num)
    delta = 0.01
    percent = 0
    sz0_move = sz0_single + delta
    sz1_move = sz1_single + delta
    while True:        
        #返回原筛网位置以上颗粒 和 平移筛网以下颗粒   计算10% 和 90%颗粒的平面位置
        flag_under = flagUnder(px_single,pz_single,sx0_single,sx1_single,sz0_move,sz1_move,-45)
        flag = flag_above * flag_under        
        # print(sum(sum(flag)),total_num)
        percent = float(sum(flag.ravel()))/total_num
        print('percent:',percent)  

        sz0_move += delta
        sz1_move += delta      
        if percent <= 0.1:
            plane1 = [sx0_single,sz1_move,sx1_single,sz0_move]
        elif percent >= 0.9:
            plane2 = [sx0_single,sz1_move,sx1_single,sz0_move]
            # # 可以用于循环画出各个时刻时的筛上颗粒和料层
            # plt.scatter(px_single*flag_above,pz_single*flag_above,color='r',marker='.',alpha=0.3)
            # plt.scatter(px_single*flag,pz_single*flag,color='g',marker='.',alpha=0.3)
            # plt.show()
            return plane1, plane2, flag
    # #画图验证这两条直线位置
    # params1 = calLine(plane1[0],plane1[1],plane1[2],plane1[3])
    # params2 = calLine(plane2[0],plane2[1],plane2[2],plane2[3])
    # print(params1,params2)
    # x = np.arange(-80,80); y1 = params1[0]*x + params1[1]; y2 = params2[0]*x + params2[1]
    # plt.plot(x,y1,'g'); plt.plot(x,y2,'r'); plt.show()


def main():

    #需要修改的地方！！
    ########################
    address_file = 'F:/self_orthogonal_experiment/data/test7'
    num_time = 11 #表示自己选取的时刻数，可选5个时刻，或7个时刻……可以通过数数据输出时停顿多少下，n-1
    z_min = -45 #防止已经筛过的颗粒计算入内
    ########################
    #颗粒数据目录
    address_particle_x = address_file+'/particle_x.csv'
    address_particle_z = address_file+'/particle_z.csv'
    address_particle_mass = address_file+'/particle_mass.csv'
    
    #筛网数据目录，筛网上小点的坐标，分别是各个时刻的坐标值 x0/z0表示最小值，x1/z1表示最大值
    address_screen_x0 = address_file+'/screen_x0.csv'
    address_screen_x1 = address_file+'/screen_x1.csv'
    address_screen_z0 = address_file+'/screen_z0.csv'
    address_screen_z1 = address_file+'/screen_z1.csv'

    #读入数据  用csv库
    data_px = loadData(address_particle_x)
    data_pz = loadData(address_particle_z)
    data_pm = loadData(address_particle_mass)
    
    data_sx0 = loadData(address_screen_x0)
    data_sx1 = loadData(address_screen_x1)
    data_sz0 = loadData(address_screen_z0)
    data_sz1 = loadData(address_screen_z1)
    print('data_sx0:{}'.format(data_sx0))

    #将颗粒的数据，分割为以单个时刻为一行的 矩阵，第一列是时刻
    data_px_m = dataSplit(data_px,num_time)
    data_pz_m = dataSplit(data_pz,num_time)
    data_pm_m = dataSplit(data_pm,num_time)

    #抽出时间列表
    time_list = data_px_m[:,0:1] #表示颗粒矩阵的所有行的第一列
    times = []
    for i in time_list:
        times.append(float(i))
    print('time list:',times)

    # #画出几个时刻下的，颗粒 筛网位置情况，查看筛网位置是否正确
    # drawPlot(data_px_m,data_pz_m,data_sx0,data_sx1,data_sz0,data_sz1)

    #将筛箱左侧颗粒 以0-1矩阵形式表示
    flag_left = flagLeft(data_px_m[:,1:],data_sx1)
    # plt.scatter(data_px_m[:,1:]*flag_left,data_pz_m[:,1:]*flag_left,marker='.',alpha=0.3)

    #将筛上颗粒(仅指左侧) 以0-1矩阵的形式表示 使用位运算&
    flag_above = flagAbove(data_px_m[:,1:],data_pz_m[:,1:],data_sx0,data_sx1,data_sz0,data_sz1)
    # plt.scatter(data_px_m[:,1:]*flag_under,data_pz_m[:,1:]*flag_under,color='g',marker='.',alpha=0.3)
    # plt.scatter(data_px_m[:,1:]*flag_above,data_pz_m[:,1:]*flag_above,color='r',marker='.',alpha=0.3)
    # plt.show()

    #统计多个时刻，筛上颗粒数目：
    # print('content of flag_above:',flag_above[-10:-5,-10:-5])
    num_particles = numParticles(flag_above)
    print('particles above the screen:\n',num_particles)

    #找第一个10%的颗粒界面d1 和 第二个10%的颗粒界面d2，返回flag_middle_m代表中间80%的料层
    flag_middle_m = []; plane1_m = []; plane2_m = []
    for i in range(num_time):
        plane1, plane2, flag_middle = moveScreen(
            data_px_m[i,1:],data_pz_m[i,1:],data_sx0[i],data_sx1[i],data_sz0[i],data_sz1[i])
        plane1_m.append(plane1); plane2_m.append(plane2); flag_middle_m.append(flag_middle)
    flag_middle_m = np.array(flag_middle_m).reshape(num_time,-1)
    print('shape of middle particles:',flag_middle_m.shape)
    
    # #画图验证中间料层：可以是(2,2) or (2,num_time//2+1)
    # fig,axes = plt.subplots(2,2)
    # for i,ax in enumerate(axes.ravel()):
    #     try:
    #         ax.scatter(data_px_m[i,1:]*flag_above[i,:],data_pz_m[i,1:]*flag_above[i,:],color='r',marker='.',alpha=0.3)
    #         ax.scatter(data_px_m[i,1:]*flag_middle_m[i,:],data_pz_m[i,1:]*flag_middle_m[i,:],color='g',marker='.',alpha=0.1)
    #         ax.set_title('time={}s'.format(time_list[i]))
    #     except:
    #         break
    # plt.show()



if __name__=='__main__':
    main()