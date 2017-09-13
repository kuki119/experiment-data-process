#unicode:utf-8
#二维的粒子群算法成功！！
#高维计算，再增加position_x3,position_x4

import random
import numpy as np 

class Bird:
    """
    speed:速度
    position:位置
    fit:适应度
    lbestposition:经历的最佳位置
    lbestfit:经历的最佳的适应度值
    """
    def __init__(self, speed, position, fit, lBestPosition, lBestFit):
        self.speed = speed
        self.position = position
        self.fit = fit
        self.lBestFit = lBestFit
        self.lBestPosition = lBestPosition

class PSO:
    """
    fitFunc:适应度函数
    birdNum:种群规模
    w:惯性权重
    c1,c2:个体学习因子，社会学习因子
    solutionSpace:解空间，列表类型：[最小值，最大值]
    """
    def __init__(self, fitFunc, birdNum, w, c1, c2, solutionSpace):#这里假设两个自变量的取值范围一样
        self.fitFunc = fitFunc
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.birds, self.best = self.initbirds(birdNum, solutionSpace)

    def initbirds(self, size, solutionSpace):
        birds = []
        for i in range(size):
            #提升自变量维度时候，在这里增添自变量个数，position_x3,position_x4
            position_x1 = random.uniform(solutionSpace[0], solutionSpace[1])  
            position_x2 = random.uniform(solutionSpace[0], solutionSpace[1])  
            position = np.array([position_x1,position_x2],'int') #'int' 是不是可以在这里替换 int和float 实现整数优化
            speed = 0
            fit = self.fitFunc(position)
            birds.append(Bird(speed, position, fit, position, fit))
        best = birds[0]
        for bird in birds:
            if bird.fit > best.fit:
                best = bird
        return birds,best

    def updateBirds(self):
        for bird in self.birds:
            # 更新速度
            bird.speed = self.w * bird.speed + self.c1 * random.random() * (bird.lBestPosition - bird.position) + self.c2 * random.random() * (self.best.position - bird.position)
            # 更新位置
            bird.position = np.array(bird.position + bird.speed,'float') #'int' 是不是可以在这里替换 int和float 实现整数优化
            # 跟新适应度
            bird.fit = self.fitFunc(bird.position)
            # 查看是否需要更新经验最优
            if bird.fit > bird.lBestFit:
                bird.lBestFit = bird.fit
                bird.lBestPosition = bird.position

    def solve(self, maxIter):
        # 只考虑了最大迭代次数，如需考虑阈值，添加判断语句就好
        for i in range(maxIter):
            # 更新粒子
            self.updateBirds()
            for bird in self.birds:
                # 查看是否需要更新全局最优
                if bird.fit > self.best.fit:
                    self.best = bird

if __name__ == '__main__':
    

    def fitFunc(x):
        x1=x[0];x2=x[1]
        return (-10*(x1+30)**2-65*(x2+40)**2+20+20*x1*x2)

    f1=PSO(fitFunc,10000,0.4,2,2,[-10,10])
    f1.solve(100)
    # print(65/(2*10))
    print('x position:',f1.best.lBestPosition)
    print('max:',f1.best.lBestFit)



