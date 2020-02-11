# ------------------------------
# 工具库：主要是计算微云响应时间,对粒子位置与速度进行检查的函数
# ------------------------------
import math
import numpy as np


# Erlang C 公式
def ErlangC(n, p):
    """
    :param n: 微云的服务器个数n
    :param p: 任务到达率/微云的服务率
    :return: ErlangC公式计算出的值
    """
    try:
        p = p + 1e-5
        L = (n * p) ** n / math.factorial(n)
        R = 1 / (1 - p)
        M = L * R
        sum_ = 0
        for k in range(n):
            sum_ += (n * p) ** k / math.factorial(k)
        return M / (sum_ + M)
    except RuntimeWarning:
        print(p)

# 对粒子的速度进行检查
def CheckSpeed(velocity, overCld, underCld, cloudlets):
    """
    :param velocity:粒子的速度
    :param overCld:过载微云序号集合
    :param underCld:不过载微云序号集合
    :param cloudlets: 微云集合
    """
    len_Vs = len(overCld)
    len_Vt = len(underCld)
    for i in range(len_Vs):
        arrRate = cloudlets[overCld[i]].arrivalRate  # 任务到达率
        for j in range(len_Vt):
            # 如果粒子的某一个分量速度超过取值范围，则设为边界值
            if velocity[i][j] > arrRate * 0.1:
                velocity[i][j] = arrRate * 0.1
            elif velocity[i][j] < -arrRate * 0.1:
                velocity[i][j] = -arrRate * 0.1


# 对粒子的位置进行检查
def CheckSolution(solution, overCld, underCld, cloudlets, flag=False):
    """
    :param solution: 粒子的解
    :param overCld: 过载微云序号集合
    :param underCld: 不过载微云序号集合
    :param cloudlets: 微云集合
    :param flag: 用来判断是否是初始化解时的粒子解的检查
    """
    len_Vs = len(overCld)
    len_Vt = len(underCld)
    # 首先检查解当中是否有负数存在，存在则置为0，保证解中的每个数都>=0
    # (不过在初始化的时候不需要取检查解中是否存在负数，因为初始化保证所有值均>=0)
    if not flag:
        for i in range(len_Vs):
            for j in range(len_Vt):
                if solution[i][j] < 0:
                    solution[i][j] = 0
    # 接下来对解的每行进行检查，防止每行值之和超过过载微云i的任务到达率
    for i in range(len_Vs):
        while solution[i, :].sum() > cloudlets[overCld[i]].arrivalRate:
            row = solution[i, :]
            row_max_index = np.argmax(row)
            # 如果该行中最大的值大于任务到达率的一半，就将其除以2
            if solution[i][row_max_index] >= cloudlets[overCld[i]].arrivalRate / 2:
                solution[i][row_max_index] = solution[i][row_max_index] / 2
            solution[i][row_max_index] -= np.round(np.random.rand(), decimals=5)
            if solution[i][row_max_index] < 0:
                solution[i][row_max_index] = 0
    # 接下来对解的每列进行检查，防止每列值之和超过不过载微云j的总任务接受率(⭐⭐⭐要考虑不过载微云本身的任务到达率)
    for j in range(len_Vt):
        arg = cloudlets[underCld[j]].serverNum * cloudlets[underCld[j]].serverRate - \
              cloudlets[underCld[j]].arrivalRate
        while solution[:, j].sum() >= arg:
            col = solution[:, j]
            col_max_index = np.argmax(col)  # 获取该列中的最大值下标
            solution[col_max_index][j] -= np.round(np.random.rand(), decimals=5)
            if solution[col_max_index][j] < 0:
                solution[col_max_index][j] = 0