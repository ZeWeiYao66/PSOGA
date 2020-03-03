# ------------------------------
# PGComponent.py: Individual类、Population类
# -----------------------------
from numpy import array, argmin
import numpy as np
from numpy.random import uniform
import copy
from PGCloudlet import Cloudlet, Cloudlets
from Utils import *
import operator


# 表征个体（即粒子）
class Individual:
    # 粒子参数初始化
    def __init__(self):
        self.solution = None  # 粒子所代表的解（也即粒子的位置）
        self.velocity = None  # 粒子的速度
        self.fitness = None  # 粒子的适应度
        self.pbest = None  # 粒子的个体极值
        self.pbestFitness = None  # 粒子的个体极值适应度
        self.responseTime = None  # 粒子的响应时间
        self.pbestResponseTime = None  # 粒子的个体极值对应的所有微云的响应时间

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
            arr_rate = cloudlets[overCld[i]].arrivalRate
            temp = uniform(0, arr_rate, size=len_Vt)
            sol.append(temp)
        # 转换成numpy.ndarray,并且精确到小数点后5位
        sol = np.round(array(sol), decimals=5)
        # 对生成的解进行检查
        self.solution = CheckSolution(sol, overCld, underCld, cloudlets)
        # 更新个体极值(⭐⭐⭐必须要用copy函数，否则修改solution属性，pbest属性也会跟着变化)
        self.pbest = self.solution.copy()
        # 更新个体的适应度值
        cloudlets_copy = copy.deepcopy(cloudlets)
        self.Calculate_fitness(overCld, underCld, cloudlets_copy, delayMatrix)
        self.pbestFitness = self.fitness
        # 更新个体极值的响应时间（需要进行赋值，否则其中一个改变另一个也会跟着改变）
        self.pbestResponseTime = self.responseTime.copy()

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
            arr_rate = cloudlets[overCld[i]].arrivalRate * 0.1
            # 速度区间取任务到达率的10%
            temp = uniform(-arr_rate, arr_rate, size=len_Vt)
            speed.append(temp)
        self.velocity = np.round(array(speed), decimals=5)

    # 计算粒子对应的适应度值
    def Calculate_fitness(self, overCld, underCld, cloudlets, delayMatrix):
        """
        :param overCld: 过载微云的序号集合
        :param underCld: 不过载微云的序号集合
        :param cloudlets: 微云集合
        :param delayMatrix: 网络延时矩阵
        :return: 所有微云的任务响应时间
        """""
        len_Vs = len(overCld)  # 过载微云序号集合的长度
        len_Vt = len(underCld)  # 不过载微云序号集合的长度
        # 1.对过载微云的任务到达率进行更新
        for oIndex in range(len_Vs):
            sol_sum_row = self.solution[oIndex, :].sum()
            cloudlets[overCld[oIndex]].arrivalRate -= sol_sum_row
        # 2.对不过载微云的任务到达率进行更新
        for uIndex in range(len_Vt):
            sol_sum_col = self.solution[:, uIndex].sum()
            cloudlets[underCld[uIndex]].arrivalRate += sol_sum_col
        # 3.对于过载微云只需要计算任务等待时间，而过载微云需要计算任务等待时间与网络延迟
        responseTime = [0 for _ in range(len_Vs + len_Vt)]
        # 4.计算过载微云的任务响应时间
        for i in range(len_Vs):
            OverWaitTime = cloudlets[overCld[i]].CalWaitTime()
            responseTime[overCld[i]] = OverWaitTime
        # 5.计算不过载微云的任务响应时间
        for j in range(len_Vt):
            # 不过载微云的任务等待时间
            UnderWaitTime = cloudlets[underCld[j]].CalWaitTime()
            # 不过载微云的总网络延时
            UnderDelayTime = 0
            for n in range(len_Vs):
                UnderDelayTime += self.solution[n][j] * delayMatrix[overCld[n]][underCld[j]]
            responseTime[underCld[j]] = UnderWaitTime + UnderDelayTime
        self.fitness = np.round(max(responseTime), decimals=5)
        self.responseTime = np.round(responseTime, decimals=5)


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
        self.responTime = None  # 全局极值对应的所有微云的响应时间

    # 初始化粒子群
    def initialize(self):
        IndCls = self.individual.__class__
        # 声明individual对象
        self.individuals = [IndCls() for _ in range(self.size)]
        # 初始化粒子的位置，并更新适应度值
        self.initSolu()
        # 更新全局极值
        fitnesses = [self.individuals[i].pbestFitness for i in range(self.size)]
        gbest_index = argmin(fitnesses)
        self.gbest = self.individuals[gbest_index].pbest.copy()
        self.gbestFitness = min(fitnesses)
        self.responTime = self.individuals[gbest_index].pbestResponseTime.copy()
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
        # 更新粒子的速度
        self.individuals[index].velocity = np.round(V_t_plus_1, decimals=5)
        # 更新位置与速度
        X_t_plus_1 = self.individuals[index].solution + self.individuals[index].velocity
        # 检查粒子位置是否符合条件
        self.individuals[index].solution = CheckSolution(X_t_plus_1, self.overCld, self.underCld,
                                                         self.cloudlets.cloudlets)

    # 计算所有个体的适应度值
    def CalculateAllFit(self):
        for k in range(self.size):
            # 对微云集合进行深复制，这样子对复制集合操作不会对原集合产生影响
            cloudlets_copy = copy.deepcopy(self.cloudlets.cloudlets)
            self.individuals[k].Calculate_fitness(self.overCld, self.underCld, cloudlets_copy, self.cloudlets.C)

    # 更新个体极值
    def update_pbest(self):
        for k in range(self.size):
            # 如果更新过的粒子的适应度值比之前好，就对个体极值进行更新
            if self.individuals[k].pbestFitness > self.individuals[k].fitness:
                self.individuals[k].pbestFitness = self.individuals[k].fitness
                self.individuals[k].pbest = self.individuals[k].solution.copy()
                self.individuals[k].pbestResponseTime = self.individuals[k].responseTime.copy()

    # 更新全局极值
    def update_gbest(self):
        # 获取粒子群的适应度值
        fitnesses = [self.individuals[i].pbestFitness for i in range(self.size)]
        # 选取适应度值最小的粒子作为全局极值
        gbest_fit = min(fitnesses)
        gbest_index = argmin(fitnesses)
        if self.gbestFitness > gbest_fit:
            self.gbestFitness = gbest_fit
            self.gbest = self.individuals[gbest_index].pbest.copy()
            self.responTime = self.individuals[gbest_index].pbestResponseTime.copy()

