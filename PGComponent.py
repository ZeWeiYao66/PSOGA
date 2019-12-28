# ------------------------------
# PGComponent.py: Individual类、Population类
# ------------------------------
import numpy as np


# 表征个体（即粒子）
class Individual:
    # 粒子参数初始化
    def __init__(self):
        self.solution = None  # 粒子所代表的解（也即粒子的位置）
        self.velocity = None  # 粒子的速度
        self.fitness = None  # 粒子的适应度
        self.pbest = None  # 粒子的个体极值

    # 初始化粒子的解
    def initSolution(self):
        pass


# 表征种群
class Population:
    def __init__(self, individual, size, w=0.8, c1=2, c2=2):
        self.individual = individual
        self.size = size            # 种群大小
        self.w = w                  # 惯性权重
        self.c1 = c1                # 个体学习因子
        self.c2 = c2                # 社会学习因子
        self.r1 = np.random.rand()  # [0,1]上的随机数
        self.r2 = np.random.rand()
        self.gbest = None           # 全局极值
        self.individuals = None     # 种群

    # 初始化粒子的解
    def initSolu(self):
        pass

    # 初始化粒子的速度
    def initVelo(self):
        pass

    # 更新粒子的位置和速度
    def update_position(self):
        pass

    # 更新个体极值
    def update_pbest(self):
        pass

    # 更新全局极值
    def update_gbest(self):
        pass