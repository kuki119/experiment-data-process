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

def getConvertMatrix(points):
#计算转换矩阵 以筛网建立坐标系 
#传入筛网的两个点
#转换矩阵 新坐标系基底在旧坐标系里的投影坐标  
#转换矩阵 [cos(a),sin(a);-sin(a),cos(a)]
    import numpy as np
    p = points
    dp = np.array([abs(p[0]-p[2]),abs(p[1]-p[3])])
    modp = np.sqrt(sum(dp**2)) #求向量的模
    sinp = dp[1]/modp
    cosp = dp[0]/modp
    con_matrix = np.matrix([[cosp,sinp],[-sinp,cosp]])
    return con_matrix

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
    dim = 0.9 #指定分类粒径
    z_min = -44
    x_max = scn_tim[2]
    ptc = ptc[:,[0,2,3]]
    kb = calLine(scn_tim)
    bool_und = (ptc[:,0]<x_max)&(ptc[:,1]>z_min)&((ptc[:,1]-ptc[:,0]*kb[0]-kb[1])<0)
    ptc_und = ptc[bool_und]
    dns = 2678  #密度2678千克每立方米
    dim = dim/1000 #将 毫米 单位化成 米
    mass_std = ((np.pi/6)*dns*dim**3)*1000 #把质量的kg单位变成g单位 分离粒径颗粒质量
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
    x_bod = -66
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
    bool_stab = (are1>max(total)*0.015)&(are2>max(total)*0.015) # 布尔矩阵，可用于稳定筛分阶段时间的取值
    id_tim_stab = np.arange(tim_len)[bool_stab] #取出稳定筛分阶段的 时刻 的标号
    return id_tim_stab[0],id_tim_stab[-1] #只返回 起始 和 截止 时刻标号

def getBedPtc(ti,ptc,scn):
############返回料层颗粒################
    import numpy as np
    x_bod = -66 #入料柱的右侧最大值
    # ti=id_stab_tims[0]
    kd = calLine(scn) 
    ptc_xz = ptc[:,[0,2]]
    bool_up = (ptc_xz[:,0]>x_bod)&(ptc_xz[:,0]<scn[2])&((ptc_xz[:,1]-(kd[0]*ptc_xz[:,0]+kd[1]))>0)
    num_up = sum(bool_up) #统计此时筛面上有多少颗粒
    ptc_up = ptc_xz[bool_up]

    #尝试使用坐标转换矩阵 
    cm = getConvertMatrix(scn)
    ptc_newup = np.array(np.dot(ptc_up,cm))
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(ptc_up[:,0],ptc_up[:,1],'.',c='k',alpha=0.3)
    # ax.plot(ptc_newup[:,0],ptc_newup[:,1],'.',c='r',alpha=0.7)
    # plt.show() #坐标转换后的展示图

    #料层颗粒 不要上下的10%颗粒 即 只要中部80%的颗粒
    deta = 0.01 #z轴方向 z_min 叠加deta 然后布尔求和，找出10% 90%的分界点
    z_min = min(ptc_newup[:,1])
    prct = 1/num_up #表征所选区域中颗粒占筛上颗粒的百分比
    z_label = np.empty(2) #存放10% 和 90%的标志位
    while prct<=0.95:
        z_min += deta
        num_und = sum(ptc_newup[:,1]<z_min)
        prct = num_und/num_up
        if 0.05<prct<=0.1:
            z_label[0] = z_min
            # print(prct) #10%位点的迭代过程
        elif 0.85<prct<=0.9:
            z_label[1] = z_min
            # print(prct) #90%位点的迭代过程
    bool_bed = (ptc_newup[:,1]>z_label[0])&(ptc_newup[:,1]<z_label[1])
    ptc_bed = np.dot(ptc_newup[bool_bed],cm.I)
    return ptc_bed

def calPorosity():
#计算松散性
    pass

def calStratification():
#计算分层状态
    pass

def calCollision():
#计算触筛概率
    pass

def calPenetrating():
#计算透筛概率 
    pass

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