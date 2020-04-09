# ------------------------------
# PGOperator.py: Mutation类
# ------------------------------
from numpy.random import shuffle
from Utils import *
import numpy as np


# 变异类
class Mutation:
    # 初始化
    def __init__(self, rate):
        self.rate = rate    # 个体变异的概率

    # 对种群的全部个体进行变异
    def mutate(self, population, overCld, underCld, cloudlets):
        for indiv in population.individuals[0:population.size]:
            # 如果随机数小于变异概率则对该个体进行变异操作
            if np.random.rand() > self.rate:
                continue
            # 进行变异
            self.mutate_individual(indiv.solution, overCld, underCld, cloudlets)
            # 对变异后粒子的解进行检查
            indiv.solution = CheckSolution(indiv.solution, overCld, underCld, cloudlets)

    # 变异操作（随机选取行和列，对这些行和列的值进行重新初始化）
    @staticmethod
    def mutate_individual(individual, overCld, underCld, cloudlets):
        """
        :param individual: 需要进行变异的个体
        :param overCld: 过载微云集合
        :param underCld: 不过载微云集合
        :param cloudlets: 微云集合
        """
        # 随机选取行和列
        len_Vs = len(overCld)  # 过载微云集合的长度
        len_Vt = len(underCld)  # 不过载微云集合的长度
        row_rand_array = np.arange(len_Vs)
        col_rand_array = np.arange(len_Vt)
        shuffle(row_rand_array)  # 对行下标进行重新排列
        shuffle(col_rand_array)  # 对列下标进行重新排列
        # 相当于对|len1|×|len2|的矩阵重新初始化
        len1 = np.random.randint(2, len_Vs) if len_Vs > 3 else len_Vs
        len2 = np.random.randint(2, len_Vt) if len_Vt > 3 else len_Vt
        # 重新初始化
        for i in range(len1):
            oIndex = row_rand_array[i]  # 选取解的第oIndex行
            arriveRate = cloudlets[overCld[oIndex]].arrivalRate  # 取对应行的过载微云的任务到达率
            for j in range(len2):
                uIndex = col_rand_array[j]
                # 重新初始化
                individual[oIndex][uIndex] = round(uniform(0, arriveRate), decimals=5)
