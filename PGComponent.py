# ------------------------------
# PGComponent.py: Individual类、Population类
# -----------------------------
import numpy as np
import copy
from PGCloudlet import Cloudlet, Cloudlets
from Utils import *


# 表征个体（即粒子）
class Individual:
    # 粒子参数初始化
    def __init__(self):
        self.solution = None  # 粒子所代表的解（也即粒子的位置）
        self.velocity = None  # 粒子的速度
        self.fitness = None  # 粒子的适应度
        self.pbest = None  # 粒子的个体极值
        self.pbestFitness = None  # 粒子的个体极值适应度

    # 初始化粒子的位置，该解为|Vs|*|Vt|的矩阵，并求适应度
    def initSolution(self, overCld, underCld, cloudlets, delayMatrix):
        """
        :param overCld: 过载微云的序号集合
        :param underCld: 不过载微云的序号集合
        :param cloudlets: 微云集合
        :param delayMatrix: 网络延时矩阵
        """
        np.set_printoptions(suppress=True)  # 取消科学计数法
        sol = []  # 随机生成的解
        len_Vs = len(overCld)  # 过载微云序号集合的长度
        len_Vt = len(underCld)  # 不过载微云序号集合的长度
        # 生成解
        '''problem2：粒子的初始化得修改，初始化的时间占用过大'''
        for i in range(len_Vs):
            temp = np.round(np.random.uniform(0, cloudlets[overCld[i]].arrivalRate, size=len_Vt), decimals=5)
            sol.append(temp)
        # 转换成numpy.ndarray
        sol = np.array(sol)
        # 对生成的解进行检查
        self.solution = CheckSolution(sol, overCld, underCld, cloudlets, True)
        # 更新个体极值
        self.pbest = sol
        # 更新个体的适应度值
        cloudlets_copy = copy.deepcopy(cloudlets)
        self.Calculate_fitness(overCld, underCld, cloudlets_copy, delayMatrix)
        self.pbestFitness = self.fitness

    # 初始化粒子的速度，为|Vs|*|Vt|的矩阵
    def initVelocity(self, overCld, underCld, cloudlets):
        """
        :param overCld: 过载微云的序号集合
        :param underCld: 不过载微云的序号集合
        :param cloudlets: 微云集合
        """
        speed = []
        len_Vs = len(overCld)  # 过载微云序号集合的长度
        len_Vt = len(underCld)  # 不过载微序号云集合的长度
        # 随机生成速度
        for i in range(len_Vs):
            # 过载微云的到达率
            arr_rate = cloudlets[overCld[i]].arrivalRate
            # 速度区间取任务到达率的10%
            temp = np.round(np.random.uniform(-arr_rate * 0.1, arr_rate * 0.1, size=len_Vt), decimals=5)
            speed.append(temp)
        self.velocity = np.array(speed)

    # 计算粒子对应的适应度值
    def Calculate_fitness(self, overCld, underCld, cloudlets, delayMatrix):
        """
        :param overCld: 过载微云的序号集合
        :param underCld: 不过载微云的序号集合
        :param cloudlets: 微云集合
        :param delayMatrix: 网络延时矩阵
        :return: 粒子的适应度值
        """""
        len_Vs = len(overCld)  # 过载微云序号集合的长度
        len_Vt = len(underCld)  # 不过载微云序号集合的长度
        # 1.对过载微云的任务到达率进行更新
        for oIndex in range(len_Vs):
            cloudlets[overCld[oIndex]].arrivalRate -= self.solution[oIndex, :].sum()
        # 2.对不过载微云的任务到达率进行更新
        for uIndex in range(len_Vt):
            cloudlets[underCld[uIndex]].arrivalRate += self.solution[:, uIndex].sum()
        # 3.对于过载微云只需要计算任务等待时间，而过载微云需要计算任务等待时间与网络延迟
        responseTime = []
        # 4.计算过载微云的任务响应时间
        for i in range(len_Vs):
            OverWaitTime = cloudlets[overCld[i]].CalWaitTime()
            responseTime.append(OverWaitTime)
        # 5.计算不过载微云的任务响应时间
        for j in range(len_Vt):
            # 不过载微云的任务等待时间
            UnderWaitTime = cloudlets[underCld[j]].CalWaitTime()
            # 不过载微云的总网络延时
            UnderDelayTime = 0
            for n in range(len_Vs):
                UnderDelayTime += self.solution[n][j] * delayMatrix[overCld[n]][underCld[j]]
            responseTime.append(UnderWaitTime + UnderDelayTime)
        pbest_fitness = np.round(max(responseTime), decimals=5)
        self.fitness = pbest_fitness
        # return pbest_fitness


# 表征种群
class Population:
    def __init__(self, individual, cloudlets, overCld, underCld, size, w=0.8, c1=2, c2=2):
        self.individual = individual  # 个体模板
        self.cloudlets = cloudlets  # 微云集合
        self.overCld = overCld  # 过载微云
        self.underCld = underCld  # 不过载微云
        self.size = size  # 种群大小
        self.w = w  # 惯性权重
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 社会学习因子
        self.r1 = np.round(np.random.rand(), decimals=5)  # [0,1]上的随机数
        self.r2 = np.round(np.random.rand(), decimals=5)
        self.gbest = None  # 全局极值
        self.gbestFitness = None  # 全局极值对应的适应度值
        self.individuals = None  # 种群

    # 初始化粒子群
    def initialize(self):
        IndCls = self.individual.__class__
        # 声明individual对象
        self.individuals = np.array([IndCls() for i in range(self.size)], dtype=IndCls)
        # 初始化粒子的位置，并更新适应度值
        self.initSolu()
        # 更新全局极值
        self.update_gbest()
        # 初始化粒子的速度
        self.initVelo()

    # 初始化粒子的解
    def initSolu(self):
        for i in range(self.size):
            self.individuals[i].initSolution(self.overCld, self.underCld, self.cloudlets.cloudlets, self.cloudlets.C)

    # 初始化粒子的速度
    def initVelo(self):
        for i in range(self.size):
            self.individuals[i].initVelocity(self.overCld, self.underCld, self.cloudlets.cloudlets)

    # 更新粒子的位置和速度
    def update_position(self, index):
        """
        :param index: 更新第index个粒子
        """
        """problem2: 速度需要考虑界限，更新粒子的解时会出现负值，要进行调整"""
        V_t_plus_1 = self.w * self.individuals[index].velocity \
                     + self.c1 * self.r1 * (self.individuals[index].pbest - self.individuals[index].solution) \
                     + self.c2 * self.r2 * (self.gbest - self.individuals[index].solution)
        # 检查粒子速度是否符合条件
        CheckSpeed(V_t_plus_1, self.overCld, self.underCld, self.cloudlets.cloudlets)
        # 更新位置与速度
        X_t_plus_1 = self.individuals[index].solution + np.round(V_t_plus_1, decimals=5)
        # 检查粒子位置是否符合条件
        self.individuals[index].solution = CheckSolution(X_t_plus_1, self.overCld, self.underCld, self.cloudlets.cloudlets)
        self.individuals[index].velocity = np.round(V_t_plus_1, decimals=5)

    # 更新个体极值
    def update_pbest(self):
        # 对微云集合进行深复制，这样子对复制集合操作不会对原集合产生影响
        for k in range(self.size):
            cloudlets_copy = copy.deepcopy(self.cloudlets.cloudlets)
            # 计算个体的适应度值
            self.individuals[k].Calculate_fitness(self.overCld, self.underCld, cloudlets_copy, self.cloudlets.C)
            # 如果更新过的粒子的适应度值比之前好，就对个体极值进行更新
            if self.individuals[k].pbestFitness > self.individuals[k].fitness:
                self.individuals[k].pbestFitness = self.individuals[k].fitness
                self.individuals[k].pbest = self.individuals[k].solution

    # 更新全局极值
    def update_gbest(self):
        # 获取粒子群的适应度值
        fitnesses = []
        for i in range(self.size):
            fitnesses.append(self.individuals[i].pbestFitness)
        # 选取适应度值最小的粒子作为全局极值
        gbest_fit = min(fitnesses)
        gbest_index = np.argmin(fitnesses)
        if self.gbest is None:
            self.gbestFitness = gbest_fit
            self.gbest = self.individuals[gbest_index].pbest
        elif self.gbestFitness > gbest_fit:
            self.gbestFitness = gbest_fit
            self.gbest = self.individuals[gbest_index].pbest
