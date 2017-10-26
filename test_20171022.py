#coding:utf-8
#2017/10/15 实现颗粒与筛网数据的预处理
#2017/10/20 实现全部功能 有待改进细节和检验

from data_preprocess import dataParticle,dataScreen
from functions import *

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
import time

#1、########原始颗粒数据###########
ptc_tim = dataParticle(path,x_name,y_name,z_name,mass_name) #返回的数据是list 以各个时刻保存 其下为数组元素
# ptc_tim[0] 第一个时刻的颗粒数据

#2、########筛网位置数据###########
scn_tim,tim_ls = dataScreen(path,x_max_name,x_min_name,z_max_name,z_min_name)
# scn_tim[0,0:2] 第一个时刻的左侧点  scn_time[0,2:] 第一个时刻的右侧点

#3、########筛分截止时刻标号###########
id_edtim = getEndTime(ptc_tim,scn_tim,tim_ls) #返回筛分结束的时刻标号

# plt.plot(ptc_tim[id_edtim][:,0],ptc_tim[id_edtim][:,1],'.',c='k') #画出颗粒分布
# plt.plot(scn_tim[id_edtim,[0,2]],scn_tim[id_edtim,[1,3]],c='r') #画出筛网位置
# plt.show()

#4、########筛分效率 单位时间筛分效率###########
eff = calScrEff(ptc_tim[id_edtim],scn_tim[id_edtim]) #结束时刻的筛分效率
unit_eff = eff/tim_ls[id_edtim] #单位时间筛分效率
print('the unit efficiency:{}'.format(unit_eff))
# tim = timEff98(tim_ls,ptc_tim,scn_tim) #可计算出达到98%最大筛分效率的时间

#5、########稳定筛分阶段 时刻标号###########
id_stab_tims = stabTim(ptc_tim,scn_tim) #可计算稳定筛分阶段的起止时刻 标号
print('the stable screening [{0}s--{1}s]'.format(tim_ls[id_stab_tims[0]],tim_ls[id_stab_tims[1]]))

#6、########指定时刻###########
ti = id_stab_tims[0]+5 #指定 计算四要素 的时刻 即四要素都是ti时刻的状态！！
# ti = 40 #指定 计算四要素 的时刻 即四要素都是ti时刻的状态！！
print('choose time:{}'.format(tim_ls[ti]))

#7、########料层颗粒数据###########
ptc_bed,bed_z = getBedPtc(ti,ptc_tim[ti],scn_tim[ti]) #返回指定时刻的料层颗粒 转换坐标后的颗粒
print('data of the particle bed:{}'.format(ptc_bed.shape))

#8、########松散系数###########
por = calPorosity(ptc_bed) #计算松散系数 传入一个时刻下的料层颗粒
print('porosity :{}'.format(por))

#给出稳定筛分阶段 多个时刻下的 松散系数
# times = np.arange(id_stab_tims[0],id_stab_tims[1],4)
# por = np.empty(len(times))
# for i,ti in enumerate(times):
# 	ptc_bed,scn_z_min,bed_z_label = getBedPtc(ti,ptc_tim[ti],scn_tim[ti]) #返回指定时刻的料层颗粒 转换坐标后的颗粒
# 	por[i] = calPorosity(ptc_bed)

#9、########分层系数###########
stra = calStratification(ptc_bed) #使用相关系数概念
print('stratification:{}'.format(stra))
# import pandas as pd 
# ti = np.arange(17,42)
# r = np.empty(len(ti))
# stra = np.empty(len(ti))
# for j,i in enumerate(ti):
# 	ptc_bed,bed_z = getBedPtc(ti,ptc_tim[i],scn_tim[i])
# 	#尝试用相关性系数 来表征分层效果
# 	# x = ptc_bed[:,2] #颗粒z轴坐标
# 	# y = calDiam(ptc_bed[:,3]) #颗粒粒径
# 	# data = np.vstack([x,y]).T
# 	# df = pd.DataFrame(data)
# 	r[j] = calStratification(ptc_bed)
# 	# stra[j] = calStratification(ptc_bed,bed_z)


# stratifi = calStratification(ptc_bed,bed_z) #计算分层沉降系数 使用沉降系数概念

#10、########触筛概率 与 透筛概率###########
pm,pt = calMeetThrou(ptc_tim[ti],scn_tim[ti]) #计算触筛概率pm 和 透筛概率pt
print('probability-meet:{}%'.format(pm*100))
print('probability-through:{}%'.format(pt*100))
#sum(ptc_scnard[:,2]<0) #即筛网附近颗粒中 筛下的颗粒

# plt.scatter(ptc_up[:,0],ptc_up[:,1],marker='.');plt.show()


#11、#######计算稳定筛分阶段所有时刻下的四要素值################
tim_id = np.arange(id_stab_tims[0],id_stab_tims[1])
por = np.empty(len(tim_id))
stra = np.empty(len(tim_id))
pm = np.empty(len(tim_id))
pt = np.empty(len(tim_id))
for i,ti in enumerate(tim_id):
	ptc_bed,bed_z = getBedPtc(ti,ptc_tim[ti],scn_tim[ti])
	por[i] = calPorosity(ptc_bed)
	stra[i] = calStratification(ptc_bed)
	pm[i],pt[i] = calMeetThrou(ptc_tim[ti],scn_tim[ti])

por_me = np.mean(por)
stra_me = np.mean(stra)
pm_me = np.mean(pm)
pt_me = np.mean(pt)