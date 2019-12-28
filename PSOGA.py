import numpy as np
import random
import scipy.stats as stats


# 粒子群遗传算法
class Psoga:
    def __init__(self, population, mutation):
        self.population = population
        self.mutation = mutation

    def run(self, overCld, underCld, gen=100):
        """
        :param overCld: 过载微云集合
        :param underCld: 不过载微云集合
        :param gen: 种群迭代数
        :return: 全局极值
        """
        # Step1：初始化粒子群的解、速度，同时设置个体极值与全局极值
        #        （注意这里需要结合微云的划分......）
        self.population.initSolu()
        self.population.initVelo()
        # Step2：进行种群的迭代
        for n in range(1, gen + 1):
            # Step3：对每个粒子的速度、位置进行更新，并进行变异操作
            for i in range(self.population.size):
                # 1).粒子群速度、位置的更新
                self.population.update_position()

                # 2).进行变异操作(需要检查变异的粒子是否符合要求)
                if np.random.rand() < self.mutation.rate:
                    self.mutation.mutate(self.population.individuals[i])

                # 3).更新个体极值
                self.population.update_pbest()
            # 4).更新全局极值
            self.population.update_gbest()

        # Step4：结束迭代，返回全局极值
        return
