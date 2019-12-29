# ------------------------------
# 工具库：主要是计算响应时间的公式函数
# ------------------------------
import math


# Erlang C 公式
def ErlangC(n, p):
    L = (n*p)**n/math.factorial(n)
    R = 1/(1-p)
    M = L*R
    sum_ = 0
    for k in range(n):
        sum_ += (n*p)**k/math.factorial(k)
    return M/(sum_+M)


# # 计算适应度
# def fitness(population, aimFunction):
#     """
#     :param population: 种群
#     :param aimFunction: 目标函数
#     :return: 种群的适应度值
#     """
#     pass

if __name__ == '__main__':
   pass