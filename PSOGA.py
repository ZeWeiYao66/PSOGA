import numpy as np
import random
import scipy.stats as stats
from Utils import *
import time


# 粒子群遗传算法
class Psoga:
    def __init__(self, population, mutation, Wmin=0.3, Wmax=0.8):
        self.population = population  # 种群
        self.mutation = mutation  # 变异操作
        self.Wmin = Wmin  # 惯性权重取值范围
        self.Wmax = Wmax

    def run(self, gen=100):
        """
        :param gen: 种群迭代数
        :return: 全局极值
        """
        # Step1：初始化粒子群的解、速度，同时设置个体极值与全局极值
        start_time1 = time.process_time()
        self.population.initialize()
        end_time1 = time.process_time()
        print('initialize time: ', end_time1 - start_time1)
        # Step2：进行种群的迭代
        start_time2 = time.process_time()
        overCld = self.population.overCld
        underCld = self.population.underCld
        cloudlets = self.population.cloudlets.cloudlets
        for n in range(1, gen + 1):
            # Step3：对每个粒子的速度、位置进行更新，并进行变异操作
            # self.population.w = self.Wmax - n * (self.Wmax - self.Wmin) / gen
            self.population.w = self.Wmax - (self.Wmax - self.Wmin) * (np.log(n) / np.log(gen))
            for index in range(self.population.size):
                # 1).粒子群速度、位置的更新
                self.population.update_position(index)
                # 2).进行变异操作(需要检查变异的粒子是否符合要求)
                if np.random.rand() < self.mutation.rate:
                    # 获取参数
                    solution = self.population.individuals[index].solution
                    # 进行变异操作
                    self.mutation.mutate(solution, overCld, underCld, cloudlets)
                    # 检查变异之后的粒子是否满足限制要求
                    self.population.individuals[index].solution = CheckSolution(solution, overCld, underCld, cloudlets)
            # 3).更新个体极值(全部粒子的速度和位置更新完之后再去计算)
            self.population.update_pbest()
            # 4).更新全局极值
            self.population.update_gbest()
        end_time2 = time.process_time()
        print('iterate time: ', end_time2 - start_time2)
        # Step4：结束迭代，返回全局极值
        print('最优解:  ', self.population.gbest)
        print('适应度值:', self.population.gbestFitness)
        return self.population.gbest, self.population.gbestFitness
