#coding:utf-8
#2017/10/8 写一个专门用于DEM颗粒数据组织的程序
#原始Excel数据 包含多个时刻 入料颗粒  各个时刻颗粒数量并不一定相同
#要求实现 数据分成n块 每块以 个数为行标号  mass,x,y,z为列标号  排成DataFrame 格式
#思路是  寻找各个时刻数据开头的字符串数据  依此来分块数据  最后去除这些字符串数据
#2017/10/10 实现数据的分块 输入数据路径 返回各个时刻下的数据 矩阵  作为函数使用 
#2017/10/19 修改 x,y,z,mass 四列数据 

def dataParticle(path,x_name,y_name,z_name,mass_name):
#返回数据结构 以时刻分块的列表 一块是n维数组
    from pandas import read_csv, concat
    import numpy as np 

    address_x = path+x_name
    address_y = path+y_name
    address_z = path+z_name
    address_mass = path+mass_name

    data_x = read_csv(address_x,skiprows=6)
    data_y = read_csv(address_y,skiprows=6)
    data_z = read_csv(address_z,skiprows=6)
    data_mass = read_csv(address_mass,skiprows=6)
    unit_list = [data_x.columns[1],data_y.columns[1],data_z.columns[1],data_mass.columns[1]]
    print(unit_list)
    data_x.columns = ['a','x']
    data_y.columns = ['a','y']
    data_z.columns = ['a','z']
    data_mass.columns = ['a','mass']
    # print(data_orig[1].head())

    data_orig = concat([data_x,data_y.y,data_z.z,data_mass.mass],axis=1)

    # print(data_orig.head())

    id_time = data_orig[data_orig.a.isin(['TIME:'])].index  #找到时刻的标号
    # print(len(id_time),id_time[0:5])

    data_times = []
    for i in range(len(id_time)):
        try:
            data_times.append(data_orig.iloc[id_time[i]+2:id_time[i+1],:].as_matrix(columns=['x','y','z','mass']))
        except:
            data_times.append(data_orig.iloc[id_time[i]+2:,:].as_matrix(columns=['x','y','z','mass']))
    # print(data_times[2].head(),'\n',data_times[2].tail())
    return data_times

def dataScreen(path,x_max_name,x_min_name,z_max_name,z_min_name):
##返回数据结构  一行为一个时刻 前两个为左侧点坐标，后两个为右侧点坐标[x_min,z_max,x_max,z_min]
    from pandas import concat, read_csv

    x_max = read_csv(path+x_max_name,skiprows=6)
    x_min = read_csv(path+x_min_name,skiprows=6)
    z_max = read_csv(path+z_max_name,skiprows=6)
    z_min = read_csv(path+z_min_name,skiprows=6)

    idx = x_max[x_max.iloc[:,0].isin(['TIME:'])].index+1
    times = x_max.iloc[idx-1,1].as_matrix()
    x_max = x_max.iloc[idx,1]
    x_min = x_min.iloc[idx,1]
    z_max = z_max.iloc[idx,1]
    z_min = z_min.iloc[idx,1]

    x_z = concat([x_min,z_max,x_max,z_min],axis=1)
    x_z_m = x_z.as_matrix()

    return x_z_m,times

def main():
    path = 'F:\data\\test_data1\\'
    x_name = 'bj03_x.csv'
    z_name = 'bj03_z.csv'
    mass_name = 'bj03_mass.csv'
    x_max_name = 'bj03_screen_x_max.csv'
    x_min_name = 'bj03_screen_x_min.csv'
    z_max_name = 'bj03_screen_z_max.csv'
    z_min_name = 'bj03_screen_z_min.csv'

    ptc_tim = dataParticle(path,x_name,z_name,mass_name) #返回的数据是list 以各个时刻保存 其下为数组元素
    # ptc_tim[0] 第一个时刻的颗粒数据
    scn_tim = dataScreen(path,x_max_name,x_min_name,z_max_name,z_min_name)
    # scn_tim[0,0:2] 第一个时刻的左侧点  scn_time[0,2:] 第一个时刻的右侧点

if __name__ == '__main__':
    main()