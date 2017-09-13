#coding=utf-8
#验证三门问题，当已知一扇门为空的前提下改与不改的获奖概率

import random
print'*'*80
print u'！！！游戏开始！！！'
print'*'*80

#循环比较，产生空门列表
def lsDoors(list_doors,win_door):
    doors = []
    for i in list_doors:
        if i != win_door:
            doors.append(i)
    #doors 里存放空门的序号
    return doors
    

#w0标记总共进行来几次实验,w1标记不改变选项的情况下的获奖次数,w2标记改变选项的情况下的获奖次数
w0,w1,w2 = 0,0,0
while True:
    #所有门的标号
    list_doors = [1,2,3]
    #随机产生一个获奖的门
    win_door = random.randint(1,3)
    loose_doors = lsDoors(list_doors,win_door) 
    choose1 = input('请输入你选择的门牌号：')
    w0 += 1
    #如果玩家第一次 选中 了正确的门
    if choose1 == win_door:
        i = random.randint(0,1)
        print(u'我可以告诉你第%d号是空门，你要不要改。'%loose_doors[i])
        #输入第二次选择的门牌号
        choose2 = input('请输入你的最终决定：')

        if choose2 == win_door:
            print u'因为你的意志坚定，你赢得了奖品！！'
            w1 += 1

        else:
            print u'很可惜，你改错啦，奖品与你失之交臂……'
    
    #如果玩家第一次 没选中 正确的门
    else:
        if choose1 == loose_doors[0]:
            print(u'我可以告诉你第%d号是空门，你要不要改。'%loose_doors[1])
        else:
            print(u'我可以告诉你第%d号是空门，你要不要改。'%loose_doors[0])

        choose2 = input('请输入你的最终决定：')

        #当玩家修改为正确的门时，w2自动加1
        if choose2 == win_door:
            print u'小子运气不错嘛，竟然让你改对了！！恭喜你赢得了奖品'
            w2 += 1
        else:
            print u'点背，什么东西都没有……'

    print (u'总共进行了%d次'%w0)
    num = input('是否想要退出游戏：0，退出；1,继续;6,显示概率。')
    if num == 0:
        break

    elif num == 6:
        w0 = float(w0)
        #p1显示不做改变时，获奖的概率
        p1 = (w1/w0)*100
        #p2显示作出改变时，获奖的概率
        p2 = (w2/w0)*100
        print (u'不改变原定选项，获奖的概率为：%f'%p1)
        print (u'改变原定选项后，获奖的概率为：%f'%p2)

    else:
        continue








