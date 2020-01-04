# ------------------------------
# PGComponent.py: Individual类、Population类
# ------------------------------
import numpy as np
import copy
from PGCloudlet import Cloudlet, Cloudlets


# 表征个体（即粒子）
class Individual:
    # 粒子参数初始化
    def __init__(self):
        self.solution = None  # 粒子所代表的解（也即粒子的位置）
        self.velocity = None  # 粒子的速度
        self.fitness = None  # 粒子的适应度
        self.pbest = None  # 粒子的个体极值

    # 初始化粒子的位置，该解为|Vs|*|Vt|的矩阵
    def initSolution(self, overCld, underCld, cloudlets):
        np.set_printoptions(suppress=True)  # 取消科学计数法
        sol = []  # 随机生成的解
        len_Vs = len(overCld)  # 过载微云集合的长度
        len_Vt = len(underCld)  # 不过载微云集合的长度
        # 生成解，并对每行进行检查，防止超过微云i的任务到达率
        for i in range(len_Vs):
            temp = np.round(np.random.uniform(0, cloudlets[overCld[i]].arrivalRate, size=len_Vt), decimals=5)
            while temp.sum() > cloudlets[overCld[i]].arrivalRate:
                temp = np.round(np.random.uniform(0, cloudlets[overCld[i]].arrivalRate, size=len_Vt), decimals=5)
            sol.append(temp)
        # 转换成numpy.ndarray
        sol = np.array(sol)
        # 对每列进行检查，防止超过微云j的总服务率(⭐⭐⭐要考虑不过载微云本身的任务到达率)
        for j in range(len_Vt):
            arg = cloudlets[underCld[j]].serverNum * cloudlets[underCld[j]].serverRate - \
                  cloudlets[underCld[j]].arrivalRate
            # 如果某列违反了条件，选取该列中的最大值进行随机减少，直到符合要求
            while sol[:, j].sum() >= arg:
                col = sol[:, j]
                col_max_index = np.argmax(col)  # 获取该列中的最大值下标
                sol[col_max_index][j] -= np.round(np.random.rand(), decimals=5)
                if sol[col_max_index][j] < 0:
                    sol[col_max_index][j] = 0
        self.solution = sol
        # 更新个体极值
        self.pbest = sol
        # print(self.solution)

    # 初始化粒子的速度，为|Vs|*|Vt|的矩阵
    def initVelocity(self, overCld, underCld, cloudlets):
        speed = []
        len_Vs = len(overCld)  # 过载微云集合的长度
        len_Vt = len(underCld)  # 不过载微云集合的长度
        # 随机生成速度
        for i in range(len_Vs):
            # 过载微云的到达率
            arr_rate = cloudlets[overCld[i]].arrivalRate
            # 速度区间取任务到达率的20%
            temp = np.round(np.random.uniform(-arr_rate * 0.2, arr_rate * 0.2, size=len_Vt), decimals=5)
            speed.append(temp)
        self.velocity = np.array(speed)
        # print(self.velocity)

    # 计算粒子对应的适应度值
    def Calculate_fitness(self, overCld, underCld, cloudlets, delayMatrix):
        len_Vs = len(overCld)  # 过载微云集合的长度
        len_Vt = len(underCld)  # 不过载微云集合的长度
        # 1.对过载微云的任务到达率进行更新
        for oIndex in range(len_Vs):
            cloudlets[overCld[oIndex]].arrivalRate -= self.solution[oIndex, :].sum()
        # 2.对不过载微云的任务到达率进行更新
        for uIndex in range(len_Vt):
            cloudlets[underCld[uIndex]].arrivalRate += self.solution[:, uIndex].sum()
        # 3.对于过载微云只需要计算任务等待时间，而过载微云需要计算任务等待时间与网络延迟
        responseTime = []
        # 计算过载微云的任务响应时间
        for i in range(len_Vs):
            OverWaitTime = cloudlets[overCld[i]].CalWaitTime()
            responseTime.append(OverWaitTime)
        # 计算不过载微云的任务响应时间
        for j in range(len_Vt):
            # 不过载微云的任务等待时间
            UnderWaitTime = cloudlets[underCld[j]].CalWaitTime()
            # 计算不过载微云的总网络延时
            UnderDelayTime = 0
            for n in range(len_Vs):
                UnderDelayTime += self.solution[n][j]*delayMatrix[overCld[n]][underCld[j]]
            responseTime.append(UnderWaitTime+UnderDelayTime)
        # print(responseTime)
        self.fitness = np.round(max(responseTime), decimals=5)
        return self.fitness


# 表征种群
class Population:
    def __init__(self, individual, cloudlets, overCld, underCld, size, w=0.8, c1=2, c2=2):
        self.individual = individual
        self.cloudlets = cloudlets
        self.overCld = overCld
        self.underCld = underCld
        self.size = size  # 种群大小
        self.w = w  # 惯性权重
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 社会学习因子
        self.r1 = np.random.rand()  # [0,1]上的随机数
        self.r2 = np.random.rand()
        self.gbest = None  # 全局极值
        self.individuals = None  # 种群

    # 初始化粒子群
    def initialize(self):
        IndCls = self.individual.__class__
        self.individuals = np.array([IndCls() for i in range(self.size)], dtype=IndCls)
        # 初始化粒子的位置
        self.initSolu()
        # 初始化粒子的速度
        self.initVelo()

    # 初始化粒子的解
    def initSolu(self):
        # cloudlets_copy = copy.deepcopy(self.cloudlets)
        for i in range(self.size):
            self.individuals[i].initSolution(self.overCld, self.underCld, self.cloudlets)

    # 初始化粒子的速度
    def initVelo(self):
        for i in range(self.size):
            self.individuals[i].initVelocity(self.overCld, self.underCld, self.cloudlets)

    # 计算个体的适应度值
    def CalFitness(self, delayMatrix):
        """
        :param delayMatrix: 延时矩阵
        :return:
        """
        # 每个粒子的适应度值
        fitnesses = []
        # 对微云集合进行深复制，这样子对复制集合操作不会对原集合产生影响
        for k in range(self.size):
            cloudlets_copy = copy.deepcopy(self.cloudlets)
            val = self.individuals[k].Calculate_fitness(self.overCld, self.underCld, cloudlets_copy, delayMatrix)
            fitnesses.append(val)
        print(fitnesses)

    # 更新粒子的位置和速度
    def update_position(self):
        pass

    # 更新个体极值
    def update_pbest(self):
        pass

    # 更新全局极值
    def update_gbest(self):
        pass