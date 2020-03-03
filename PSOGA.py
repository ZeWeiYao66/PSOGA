import numpy as np
import random
import scipy.stats as stats
from Utils import *
import time
import operator
from math import pow


# 粒子群遗传算法
class Psoga:
    def __init__(self, population, mutation, crossover, Wmin=0.3, Wmax=0.8):
        """
        :param population: 粒子群
        :param mutation: 变异操作
        :param crossover: 交叉操作
        :param Wmin: 惯性权重下限
        :param Wmax: 惯性权重上限
        """
        self.population = population  # 种群
        self.mutation = mutation  # 变异操作
        self.crossover = crossover  # 交叉操作
        self.Wmin = Wmin  # 惯性权重取值范围
        self.Wmax = Wmax

    def run(self, gen=100):
        """
        :param gen: 种群迭代数
        :return: 全局极值
        """
        # Step1：初始化粒子群的解、速度，同时设置个体极值与全局极值
        start_time1 = time.time()
        self.population.initialize()
        end_time1 = time.time()
        print('initialize time: ', end_time1 - start_time1)
        # Step2：进行种群的迭代
        start_time2 = time.time()
        size = self.population.size
        overCld = self.population.overCld
        underCld = self.population.underCld
        cloudlets = self.population.cloudlets.cloudlets
        # 比较函数，根据键‘fitness’属性对对象数组进行排序
        compare_fun = operator.attrgetter('fitness')
        for n in range(1, gen + 1):
            # Step3：对所有粒子进行速度、位置的更新，然后根据适应度值排序，将粒子群分成三个部分，第一部分不动
            #        第二部分进行交叉操作，第三部分进行变异操作。
            self.population.w = self.Wmax - n * (self.Wmax - self.Wmin) / gen
            # self.population.w = self.Wmax - (self.Wmax - self.Wmin) * (np.log(n) / np.log(gen))
            # self.population.w = (self.Wmax - self.Wmin) * pow(n / gen, 2) + (self.Wmax - self.Wmin) * (
            #             2 * n / gen) + self.Wmax
            # 根据适应度值对个体进行排序，从小到大
            self.population.individuals.sort(key=compare_fun)
            # 1).进行交叉操作
            self.crossover.cross(self.population)
            # 2).进行变异操作(需要检查变异的粒子是否符合要求)
            self.mutation.mutate(self.population, overCld, underCld, cloudlets)
            # 计算个体的适应度值
            self.population.CalculateAllFit()
            # 3).更新个体极值(全部粒子的速度和位置更新完之后再去计算)
            self.population.update_pbest()
            # 4).更新全局极值
            self.population.update_gbest()
            # 更新粒子的位置、速度
            for index in range(size):
                self.population.update_position(index)
            self.population.CalculateAllFit()
            # 根据适应度值对个体进行排序，从小到大
            self.population.individuals.sort(key=compare_fun)
            # 1).进行交叉操作
            self.crossover.cross(self.population)
            # 2).进行变异操作(需要检查变异的粒子是否符合要求)
            self.mutation.mutate(self.population, overCld, underCld, cloudlets)
            # 计算个体的适应度值
            self.population.CalculateAllFit()
            # 3).更新个体极值(全部粒子的速度和位置更新完之后再去计算)
            self.population.update_pbest()
            # 4).更新全局极值
            self.population.update_gbest()
        end_time2 = time.time()
        print('iterate time: ', end_time2 - start_time2)
        # Step4：结束迭代，返回全局极值
        print('最优解:  ', self.population.gbest)
        print('适应度值:', self.population.gbestFitness)
        print('响应时间:', self.population.responTime)
        return self.population.gbest, self.population.gbestFitness
