#coding:utf-8
#2017/10/15 实现颗粒与筛网数据的预处理 

from data_preprocess_backup2 import dataParticle,dataScreen
from functions_backup2 import *

def calPorosity():
#计算松散性
    pass

path = 'F:\data\\test_data1\\'
x_name = 'bj03_x.csv'
y_name = 'bj03_y.csv'
z_name = 'bj03_z.csv'
mass_name = 'bj03_mass.csv'
x_max_name = 'bj03_screen_x_max.csv'
x_min_name = 'bj03_screen_x_min.csv'
z_max_name = 'bj03_screen_z_max.csv'
z_min_name = 'bj03_screen_z_min.csv'

import matplotlib.pyplot as plt
import numpy as np

ptc_tim = dataParticle(path,x_name,y_name,z_name,mass_name) #返回的数据是list 以各个时刻保存 其下为数组元素
# ptc_tim[0] 第一个时刻的颗粒数据
scn_tim,tim_ls = dataScreen(path,x_max_name,x_min_name,z_max_name,z_min_name)
# scn_tim[0,0:2] 第一个时刻的左侧点  scn_time[0,2:] 第一个时刻的右侧点

id_edtim = getEndTime(ptc_tim,scn_tim,tim_ls) #返回筛分结束的时刻标号

# plt.plot(ptc_tim[id_edtim][:,0],ptc_tim[id_edtim][:,1],'.',c='k') #画出颗粒分布
# plt.plot(scn_tim[id_edtim,[0,2]],scn_tim[id_edtim,[1,3]],c='r') #画出筛网位置
# plt.show()

eff = calScrEff(ptc_tim[id_edtim],scn_tim[id_edtim]) #结束时刻的筛分效率

# tim = timEff98(tim_ls,ptc_tim,scn_tim) #可计算出达到98%最大筛分效率的时间

id_stab_tims = stabTim(ptc_tim,scn_tim) #可计算稳定筛分阶段的起止时刻 标号

ti = 20
ptc_bed = getBedPtc(ti,ptc_tim[ti],scn_tim[ti]) #返回指定时刻的料层颗粒

###############计算松散系数##############



# plt.scatter(ptc_up[:,0],ptc_up[:,1],marker='.');plt.show()
