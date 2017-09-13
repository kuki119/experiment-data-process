#coding: utf-8
#差分进化算法 可以对自变量增加取值范围限制
#根据需要改自变量个数与其取值范围，即 n x_l x_u

import numpy as np 
'''
n 自变量个数； m_size 粒子个数；f 缩放因子；cr 交叉概率；
x_l,x_u 分别设定了几个自变量的最小值与最大值
'''

def de(n=2,m_size=50,f=0.5,cr=0.3,iterate_times=500,x_l=np.array([0,1]),x_u=np.array([5,6])):
    x_all = np.zeros((iterate_times,m_size,n))
    for i in range(m_size):
        x_all[0][i] = x_l + np.random.random()*(x_u-x_l)
    print('差分进化算法初始化完成')
    print('寻优参数个数为：',n,'优化区间分别为：',x_l,x_u)
    for g in range(iterate_times-1):
        print('第',g,'代')
        for i in range(m_size):
            x_g_without_i = np.delete(x_all[g],i,0)
            np.random.shuffle(x_g_without_i) #shuffle把该序列中元素打乱

            h_i = x_g_without_i[1] + f*( x_g_without_i[2] - x_g_without_i[3])
            #变异操作后，h_i个体可能会超过上下限区间，为了保证在区间以内，对超过区间外的值赋予相邻的边界值
            # h_i = [h_i[item] if h_i[item] < x_u[item] else x_u[item] for item in range(n)]
            # h_i = [h_i[item] if h_i[item] > x_l[item] else x_l[item] for item in range(n)]
            print(h_i)

            #交叉操作，对变异后的个体，根据随机数与交叉阀值确定最后的个体
            v_i = np.array([x_all[g][i][j] if (np.random.random() > cr) else h_i[j] for j in range(n)])

            #选择  更改大于小于号，改变求最大值或最小值  小于号求最大值！！
            if evaluate_func(x_all[g][i]) < evaluate_func(v_i):  
                x_all[g+1][i] = v_i
            else:
                x_all[g+1][i] = x_all[g][i]

    evaluate_result = [evaluate_func(x_all[iterate_times-1][i]) for i in range(m_size)]
    label = np.argmax(evaluate_result) #沿着指定轴，返回最大值的标号
    best_x_i = x_all[iterate_times-1][label] 
    print('evaluate_result:',evaluate_result)
    print(label,evaluate_result[label])
    print('best_x_i:',best_x_i)

def evaluate_func(x):
    x1 = x[0]
    x2 = x[1]
    return (-10*(x1+30)**2-65*(x2+40)**2+20+20*x1*x2)

if __name__ == '__main__':
    de()