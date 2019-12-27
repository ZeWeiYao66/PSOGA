import numpy as np
import random
import scipy.stats as stats

# 粒子群遗传算法
class Psoga:
    def __init__(self, population, mutate):
        self.population = population
        self.mutate = mutate

    def run(self, gen=100):
        """Step1：初始化粒子群的解"""
        self.population.initialize()
        """Step2：计算适应值并不断迭代"""
        for n in range(1,gen+1):
            # 1.计算适应值
            pass

            # 2.粒子群速度、位置的更新
            pass

            # 3.进行变异操作
            pass

            # 4.更新个体极值与全局极值
            pass

        # 5.结束迭代，返回全局极值
        return
