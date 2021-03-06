#unicode:utf-8
#一维的粒子群算法成功！！
#怎么实现高维搜索？？


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
        self.lBestFit = lBestPosition
        self.lBestPosition = lBestFit

import random

class PSO:
    """
    fitFunc:适应度函数
    birdNum:种群规模
    w:惯性权重
    c1,c2:个体学习因子，社会学习因子
    solutionSpace:解空间，列表类型：[最小值，最大值]
    """
    def __init__(self, fitFunc, birdNum, w, c1, c2, solutionSpace):
        self.fitFunc = fitFunc
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.birds, self.best = self.initbirds(birdNum, solutionSpace)

    def initbirds(self, size, solutionSpace):
        birds = []
        for i in range(size):
            position = random.uniform(solutionSpace[0], solutionSpace[1])  #？？这里只返回一个随机数，适应度函数里只有一个未知数？？
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
            bird.position = bird.position + bird.speed
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
        return (-10*x**2-65*x+20)

    f1=PSO(fitFunc,100,0.4,2,2,[-100,100])
    f1.solve(100)
    print(65/(2*10))
    print('x position:',f1.best.lBestPosition)
    print('max:',f1.best.lBestFit)



