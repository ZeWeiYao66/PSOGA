# ------------------------------
# PGOperator.py: Mutation类
# ------------------------------
from numpy.random import uniform, permutation, shuffle
from numpy import round
from Utils import *
import copy
import numpy as np


# 选择类
class Selection:
    pass


# 交叉类
class CrossOver:
    @staticmethod
    # 从1/3到2/3
    def cross(population):
        num = int(population.size / 3)
        # 重新排列
        random_population = permutation(population.individuals[0:num]).tolist()
        i = num
        for individual_a, individual_b in zip(population.individuals, random_population):
            population.individuals[i].solution = np.round((individual_a.solution + individual_b.solution) / 2,
                                                          decimals=5)
            # population.individuals[i].solution = CheckSolution(solution, population.overCld, population.underCld,
            #                                                    population.cloudlets.cloudlets)
            i += 1


# 变异类
class Mutation:
    # 初始化
    def __init__(self, rate):
        """
        :param rate:变异概率
        """
        self.rate = rate

    # 对种群的后1/3个体进行变异
    def mutate(self, population, overCld, underCld, cloudlets):
        num = int(population.size / 3)
        for indiv in population.individuals[num * 2:population.size]:
            if np.random.rand() > self.rate:
                continue
            # 进行变异
            self.mutate_individual(indiv.solution, overCld, underCld, cloudlets)
            # 进行检查
            indiv.solution = CheckSolution(indiv.solution, overCld, underCld, cloudlets)

    # 变异操作（随机选取行和列，对这些行和列的值进行重新初始化），从2/3到3/3
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
        # len1 = np.random.randint(2, 20) if (len_Vs > 20) else (np.random.randint(2, len_Vs) if (len_Vs > 2) else
        # len_Vs)
        row_rand_array = row_rand_array[:len1]
        len2 = np.random.randint(2, len_Vt) if len_Vt > 3 else len_Vt
        # len2 = np.random.randint(2, 20) if (len_Vt > 20) else (np.random.randint(2, len_Vt) if (len_Vt > 2) else
        # len_Vt)
        col_rand_array = col_rand_array[:len2]
        # 重新初始化
        for i in range(len1):
            oIndex = row_rand_array[i]  # 选取解的第oIndex行
            arriveRate = cloudlets[overCld[oIndex]].arrivalRate  # 取对应行的过载微云的任务到达率
            for j in range(len2):
                uIndex = col_rand_array[j]
                # 重新初始化
                individual[oIndex][uIndex] = round(uniform(0, arriveRate), decimals=5)
