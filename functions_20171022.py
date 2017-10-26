#coding:utf-8
#2017/10/19 修改 x,y,z,mass 四列数据 

def calLine(points):
#计算二维直线参数
#传入两个点坐标的数组[x_min,z_max,x_max,z_min] 返回直线的两个参数
#使用 np.linalg.solve 计算
    import numpy as np
    p = points
    a = np.array([[p[0],1],[p[2],1]])
    b = np.array([p[1],p[3]])

    params = np.linalg.solve(a,b)
    return params 

def getEndTime(ptc_tim,scn_tim,tim_ls):
#计算筛分结束时刻  1、以筛面上颗粒数量减少率或最大颗粒数量的10%来定；2、以筛分效率稳定时刻来定
###########方式1：###########
    import numpy as np
    tim_len = len(ptc_tim)
    num_ptc = np.ones(tim_len)
    for ti in range(tim_len):
    #统计某一时刻  筛面上的颗粒数量 num_ptc
        # ti = 0 #指定时刻
        kd = calLine(scn_tim[ti]) 
        ptc_xz = ptc_tim[ti][:,[0,2]]
        bool_up = (ptc_xz[:,1]-(kd[0]*ptc_xz[:,0]+kd[1]))>0
        num_ptc[ti] = sum(bool_up) #得出筛面上颗粒总数
    id_edt = sum(num_ptc>max(num_ptc)*0.02)  #以最后一个大于最大颗粒数2%的时刻为结束时刻
    print('there still has {} on the deck when time={}'.format(num_ptc[id_edt-1],tim_ls[id_edt-1]))
    #注意！！要想取出对应时刻，需要用 times(id_edt-1)
    # import matplotlib.pyplot as plt 
    # plt.plot(np.arange(tim_len),num_ptc) #画出该实验下 各个时刻时筛面上颗粒数量变化
    # plt.show()
########方式2：#########
    # #筛分效率的变化趋势 判断筛分截止时刻
    # tim_len = len(tim_ls) 
    # eff = np.ones(tim_len)
    # for i in range(tim_len):
    #     eff[i] = calScrEff(ptc_tim[i],scn_tim[i])
    # # plt.plot(tim_ls,eff);plt.show()
    # id_ed_eff = sum(eff < max(eff)*0.98)
    return id_edt-1

def calScrEff(ptc,scn_tim):
#计算单位时间筛分效率  传入一个时刻下的 颗粒数据 和 筛网数据
    import numpy as np
    dim = 0.9 #指定分离粒径
    z_min = -44 #剔除掉已经筛过的颗粒， 可能不同的实验设置会出错！！！
    x_max = scn_tim[2]
    ptc = ptc[:,[0,2,3]]
    kb = calLine(scn_tim)
    bool_und = (ptc[:,0]<x_max)&(ptc[:,1]>z_min)&((ptc[:,1]-ptc[:,0]*kb[0]-kb[1])<0)
    ptc_und = ptc[bool_und]    
    mass_std = calStdMass(0.9) #传入指定直径返回相应的球形颗粒质量
    bool_all_sml = ptc[:,2]<mass_std
    bool_und_sml = ptc_und[:,2]<mass_std
    eff_sml = sum(ptc_und[bool_und_sml,2])/sum(ptc[bool_all_sml,2])
    eff_lag = sum(ptc_und[~bool_und_sml,2])/sum(ptc[~bool_all_sml,2])
    eff = eff_sml - eff_lag

    # plt.plot(ptc[bool_all_sml,0],ptc[bool_all_sml,1],'.',c='k',alpha=0.3) #画小颗粒分布
    # plt.plot(ptc[~bool_all_sml,0],ptc[~bool_all_sml,1],'.',c='g',alpha=0.1) #画大颗粒分布
    # plt.show() #验证图
    # return eff,eff_sml,eff_lag
    return eff

def stabTim(ptc_tim,scn_tim):
########如何返回稳定筛分 时间段########
#获得入料柱 直接设定 x=-66 看之后其他实验是否适合再改 or x_max z轴最大的1%的颗粒的x最大值？？
    # plt.scatter(ptc_tim[0][:,0],ptc_tim[0][:,1],marker='.');plt.show()
    #筛上颗粒 area1:x1>-66 x1<-56; area2:x2>scn_x_max-10 x2<scn_x_max; 
    #在这两个区域内分别有(0.015)筛上颗粒的颗粒数量  根据后续实验 可能 要改变0.015取值
    import numpy as np
    x_bod = -66  #去除料柱！！！可能出错 ！！
    tim_len = len(ptc_tim)
    are1 = np.empty(tim_len);are2 = np.empty(tim_len);total = np.empty(tim_len) 
    for ti in range(tim_len):
        ptc_xz = ptc_tim[ti][:,[0,2]]
        kb = calLine(scn_tim[ti])
        bool_up = (ptc_xz[:,0]>x_bod)&(ptc_xz[:,0]<scn_tim[ti,2])&((ptc_xz[:,1]-ptc_xz[:,0]*kb[0]-kb[1])>0)
        num_ptc_up = sum(bool_up)
        ptc_up = ptc_xz[bool_up]
        num_ptc_are1 = sum((ptc_up[:,0]>x_bod)&(ptc_up[:,0]<x_bod+10)) 
        num_ptc_are2 = sum((ptc_up[:,0]>scn_tim[ti,2]-10)&(ptc_up[:,0]<scn_tim[ti,2]))
        are1[ti] = num_ptc_are1
        are2[ti] = num_ptc_are2
        total[ti] = num_ptc_up
    bool_stab = (are1>max(total)*0.03)&(are2>max(total)*0.03) # 布尔矩阵，可用于稳定筛分阶段时间的取值
    id_tim_stab = np.arange(tim_len)[bool_stab] #取出稳定筛分阶段的 时刻 的标号
    return id_tim_stab[0],id_tim_stab[-1] #只返回 起始 和 截止 时刻标号

def getConvertMatrix(points):
#计算转换矩阵 以筛网建立坐标系 
#传入筛网的两个点
#转换矩阵 新坐标系基底在旧坐标系里的投影坐标  
#转换矩阵 [cos(a),sin(a);-sin(a),cos(a)]  二维……
    import numpy as np
    p = points
    dp = np.array([abs(p[0]-p[2]),abs(p[1]-p[3])])
    modp = np.sqrt(sum(dp**2)) #求向量的模
    sinp = dp[1]/modp
    cosp = dp[0]/modp
    #构造 二维转换矩阵 和 四维转换矩阵
    con_mat2d = np.array([[cosp,-sinp],[sinp,cosp]]).T #二维转换矩阵
    # con_matrix = np.array([[cosp,0,-sinp],[0,1,0],[sinp,0,cosp]]).T #三维转换矩阵 绕y轴转p角度
    con_mat4d = np.array([[cosp,0,-sinp,0],[0,1,0,0],[sinp,0,cosp,0],[0,0,0,1]]).T #四维转换矩阵  
    
    return con_mat2d,con_mat4d

def getBedPtc(ti,ptc,scn):
############返回料层颗粒################
    import numpy as np
    x_bod = -66 #入料柱的右侧最大值
    # ti=id_stab_tims[0]
    kd = calLine(scn) 
    ptc_xyz = ptc[:,0:3]
    bool_up = (ptc_xyz[:,0]>x_bod)&(ptc_xyz[:,0]<scn[2])&((ptc_xyz[:,2]-(kd[0]*ptc_xyz[:,0]+kd[1]))>0)
    num_up = sum(bool_up) #统计此时筛面上有多少颗粒
    ptc_up = ptc[bool_up]

    #尝试使用坐标转换矩阵 
    cm2,cm4 = getConvertMatrix(scn)
    ptc_newup = np.dot(ptc_up,cm4)
    # import matplotlib.pyplot as plt 
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(ptc_up[:,0],ptc_up[:,2],'.',c='k',alpha=0.3)
    # ax.plot(ptc_newup[:,0],ptc_newup[:,2],'.',c='r',alpha=0.7)
    # plt.show() #坐标转换后的展示图

    #料层颗粒 不要上下的10%颗粒 即 只要中部80%的颗粒
    deta = 0.01 #z轴方向 z_min 叠加deta 然后布尔求和，找出10% 90%的分界点
    z_min = min(ptc_newup[:,2])
    dz_min = z_min
    prct = 1/num_up #表征所选区域中颗粒占筛上颗粒的百分比
    z_label = np.empty(2) #存放10% 和 90%的标志位
    while prct<=0.95:
        dz_min += deta
        num_und = sum(ptc_newup[:,2]<dz_min)
        prct = num_und/num_up
        if 0.01<prct<=0.05: #调节料层颗粒 占 筛上全部颗粒的 百分比
            z_label[0] = dz_min
            # print(prct) #10%位点的迭代过程
        elif 0.9<prct<=0.95:  #调节料层颗粒 占 筛上全部颗粒的 百分比
            z_label[1] = dz_min
            # print(prct) #90%位点的迭代过程
    bool_bed = (ptc_newup[:,2]>z_label[0])&(ptc_newup[:,2]<z_label[1])
    ptc_bed = ptc_newup[bool_bed]
    return ptc_bed,z_label #返回料层颗粒 筛面z坐标 料层底面z坐标

def calPorosity(ptc_bed):
###############计算松散系数##############
#1、计算平均距离矩阵； 2、计算料层颗粒的平均直径； 3、计算松散系数
    import numpy as np
    num_bed = len(ptc_bed)
    dist = np.empty(num_bed*num_bed).reshape(num_bed,-1)
    for i,pi in enumerate(ptc_bed):
        for j,pj in enumerate(ptc_bed):
            dist[i,j] = np.sqrt((pi[0]-pj[0])**2+(pi[1]-pj[1])**2) #是个对称阵，之后可以尝试优化算法

    num_dist = (num_bed*(num_bed-1))/2
    dist_ave = sum(sum(dist))/num_dist
    mass = ptc_bed[:,3]
    diams = calDiam(mass)
    diams_ave = sum(diams)/num_bed
    por = dist_ave/diams_ave #松散系数 
    return por

def calStratification(ptc_bed): #使用沉降系数的概念时，需要加bed_z
#############计算沉降系数##############
#1、分别计算大小颗粒分层沉降系数  2、相减得到沉降差 
    import numpy as np
    # mass_std = calStdMass(0.9)
    # bool_sml = ptc_bed[:,3]<=mass_std 
    # num_all = len(ptc_bed) #料层中颗粒个数
    # num_sml = sum(bool_sml) #料层中小颗粒个数
    # ptc_sml = ptc_bed[bool_sml]
    # ptc_lag = ptc_bed[~bool_sml]
    # dist_all_ave = sum(ptc_bed[:,2]-bed_z[0])/num_all #全部颗粒距离 料层底部距离 的均值
    # dist_sml_ave = sum(ptc_sml[:,2]-bed_z[0])/num_sml
    # dist_lag_ave = sum(ptc_lag[:,2]-bed_z[0])/(num_all-num_sml)

    # strat_sml = dist_sml_ave/dist_all_ave #小颗粒的分层沉降系数
    # strat_lag = dist_lag_ave/dist_all_ave #大颗粒的分层沉降系数
    # # dstrat = strat_lag - strat_sml

    # band = bed_z[1] - bed_z[0] #料层的宽度
    # strat = dist_sml_ave/band #尝试用小颗粒平均位置与料层宽度之比表示 越小越好

###############料层Z与料层中颗粒粒径 的相关关系  r越大 分层效果越好#########
    import pandas as pd 
    x = ptc_bed[:,2] #颗粒z轴坐标
    y = calDiam(ptc_bed[:,3]) #颗粒粒径
    data = np.vstack([x,y]).T
    df = pd.DataFrame(data)
    r = df.corr().iloc[0,1] #给出z坐标与颗粒粒径的相关关系
    # #计算 x y 之间的相关性 与 斜率 推测分层情况
    # ###看一下料层里 沿着z轴 颗粒粒径 的分布情况
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(x,y,'.')
    return r

def calMeetThrou(ptc,scn):
######################计算触网概率 和 透筛概率###################
#1、对所有颗粒与筛网坐标进行坐标系转换；2、取出筛网上下 最大颗粒直径 区间内的所有颗粒 
#3、对这些颗粒的z坐标减去筛网z坐标 然后加或减自己的直径 前后两z坐标相乘 取出此时z坐标为负的颗粒 即为触网颗粒
#4、计算触筛颗粒中小颗粒的质量 与 筛面上所有小颗粒的质量 之比
    import numpy as np
# ptc = ptc_tim[ti]
# scn = scn_tim[ti]
    cm2d,cm4d = getConvertMatrix(scn)
    scn_new1 = np.dot(scn[0:2],cm2d)
    scn_new2 = np.dot(scn[2:],cm2d)
    bound = np.dot(np.array([-66,scn[1]]),cm2d)
    scn_z = scn_new1[1] # 筛网z坐标
    ptc_new = np.dot(ptc,cm4d)
    ptc_diam = calDiam(ptc[:,3])
    max_diam = max(ptc_diam) #所有颗粒中的最大颗粒直径
    ptc_new[:,2] = ptc_new[:,2] - scn_z # 所有颗粒z坐标整体下移 scn_z  即 将筛网视为 0
    ptc_scnard = ptc_new[(ptc_new[:,2]>-max_diam)&(ptc_new[:,2]<max_diam)
    &(ptc_new[:,0]<scn_new2[0])&(ptc_new[:,0]>bound[0])]
    ptc_scnard_diam = calDiam(ptc_scnard[:,3])
    bool_und = ptc_scnard[:,2]<0 #标记出筛面下的颗粒
    ptc_scnard_mv = np.zeros(len(ptc_scnard[:,2]))
    #把筛网附近颗粒中 筛网以下的颗粒上移自身直径个单位  筛网以上的颗粒下移自身直径个单位
    ptc_scnard_mv[bool_und] = ptc_scnard[bool_und,2] + ptc_scnard_diam[bool_und]
    ptc_scnard_mv[~bool_und] = ptc_scnard[~bool_und,2] - ptc_scnard_diam[~bool_und]
    ptc_mvlabel = ptc_scnard[:,2] * ptc_scnard_mv  
    bool_met = ptc_mvlabel<0 #移动前后z坐标相乘之后 z坐标值小于零 即说明该颗粒属于触网颗粒
    ptc_met = ptc_scnard[bool_met] # 触网颗粒！！！可用于透筛概率计算
    mass_std = calStdMass(0.9)
    ptc_met_sml = ptc_met[ptc_met[:,3]<mass_std] #触筛颗粒中 的 小颗粒
    ptc_up = ptc_new[ptc_new[:,2]>0]
    ptc_up_sml = ptc_up[ptc_up[:,3]<mass_std]  # 筛上颗粒中的小颗粒
    ptc_thr = ptc_met[ptc_met[:,2]<0] #透筛颗粒！！penetrated particle 

    pm = sum(ptc_met_sml[:,3])/sum(ptc_up_sml[:,3]) #触网概率！！
    pt = sum(ptc_thr[:,3])/sum(ptc_met[:,3]) #透筛概率！！
    return pm,pt

def timEff98(tim_ls,ptc_tim,scn_tim): 
#计算达到最大筛分效率98%的时刻  表征一种快慢  有没有用？？
    tim_len = len(tim_ls) 
    eff = np.ones(tim_len)
    for i in range(tim_len):
        eff[i] = calScrEff(ptc_tim[i],scn_tim[i])
    # plt.plot(tim_ls,eff);plt.show()
    id_eff = sum(eff < max(eff)*0.98)
    tim = tim_ls[id_eff]
    return tim

def calDiam(mass):
#传入颗粒质量 返回球形颗粒直径
    import numpy as np
    dns = 2678 #密度2678千克每立方米
    mass = mass/1000 #将 g 单位化成 kg
    c = 6/(dns*np.pi)
    diam = (mass*c)**(1/3)
    diam = diam*1000 #将米 转成 毫米
    return diam

def calStdMass(diam):
#传入目标 直径  返回 该直径所对应的质量
# diam = 0.9 #指定分类粒径
    import numpy as np
    dns = 2678  #密度2678千克每立方米
    diam = diam/1000 #将 毫米 单位化成 米
    mass = ((np.pi/6)*dns*diam**3)*1000 #把质量的kg单位变成g单位 分离粒径颗粒质量
    return mass
