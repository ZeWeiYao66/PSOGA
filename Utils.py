# ------------------------------
# 工具库：主要是计算微云响应时间,对粒子位置与速度进行检查的函数
# ------------------------------
import math
import numpy as np
from math import factorial
from numpy import round, maximum, argmax
from numpy.random import uniform

# Erlang C 公式
def ErlangC(n, p):
    """
    :param n: 微云的服务器个数n
    :param p: 任务到达率/微云的服务率
    :return: ErlangC公式计算出的值
    """
    # 给p加上一个很小的数，避免出现除0的现象
    p = p + 1e-5
    temp = n * p
    L = fastPower(temp, n) / factorial(n)
    R = 1 / (1 - p)
    M = L * R
    sum_ = 0
    for k in range(n):
        sum_ += fastPower(temp, k) / factorial(k)
    return M / (sum_ + M)


# 计算n次幂
def fastPower(base, power):
    """
    :param base:基数
    :param power: 指数
    :return:
    """
    result = 1
    while power > 0:
        # 如果指数为奇数
        if power & 1:
            result = result * base
        # 此处等价于power /= 2
        power >>= 1
        base = base * base
    return result


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
        arrRate = cloudlets[overCld[i]].arrivalRate * 0.1  # 任务到达率
        for j in range(len_Vt):
            # 如果粒子的某一个分量速度超过取值范围，则设为边界值
            if velocity[i][j] > arrRate:
                velocity[i][j] = arrRate
            elif velocity[i][j] < -arrRate:
                velocity[i][j] = -arrRate

def CheckSolution(solution, overCld, underCld, cloudlets):
    """
    :param solution: 粒子的解
    :param overCld: 过载微云序号集合
    :param underCld: 不过载微云序号集合
    :param cloudlets: 微云集合
    """
    len_Vs = len(overCld)
    len_Vt = len(underCld)
    # 首先检查解当中是否有负数存在，存在则置为0，保证解中的每个数都>=0
    # new_solution = np.maximum(solution, 0)
    new_solution = np.where(solution > 0, solution, 0)
    # 接下来对解的每行进行检查，防止每行值之和超过过载微云i的任务到达率
    for i in range(len_Vs):
        # 微云i的迁出的任务流之和
        cloud_arr_rate = cloudlets[overCld[i]].arrivalRate
        arr_sum_row = new_solution[i, :].sum()
        while arr_sum_row > cloud_arr_rate:
            # 获得超过的任务量差值
            diff_value_1 = arr_sum_row - cloud_arr_rate
            if diff_value_1 < 0.3:
                # 获取该行中最大值的下标
                row_max_index = np.argmax(new_solution[i, :])
                if new_solution[i][row_max_index] >= diff_value_1:
                    new_solution[i][row_max_index] -= diff_value_1
                    arr_sum_row -= diff_value_1
                    continue
            # 正数的个数,因为True相当于1，可以直接求和代表个数
            pos_num_1 = (new_solution[i, :] > 0).sum()
            num_row = np.round(diff_value_1 / pos_num_1, decimals=5)
            # 将该行的每个数都减去一个值，直到符合要求（⭐⭐⭐注意除数要为>0的个数，不然会出现循环，因为对为0的数减少，之后又置为0，该数就应该不要算进去）
            new_solution[i, :] -= num_row
            # 相减出现负值的话，将负值置为0
            new_solution[i, :] = new_solution[i, :] * (new_solution[i, :] > 0)
            arr_sum_row = new_solution[i, :].sum()
    # 接下来对解的每列进行检查，防止每列值之和超过不过载微云j的总任务接受率(⭐⭐⭐要考虑不过载微云本身的任务到达率)
    for j in range(len_Vt):
        arg = cloudlets[underCld[j]].serverNum * cloudlets[underCld[j]].serverRate - \
              cloudlets[underCld[j]].arrivalRate - 1e-3
        arr_sum_col = new_solution[:, j].sum()
        # 注意：这里的条件是sum>=arg，与上面条件不一样，要有所修改
        while arr_sum_col > arg:
            # 获得超过的任务量差值
            diff_value_2 = arr_sum_col - arg
            if diff_value_2 < 0.3:
                # 获取该行中最大值的下标
                col_max_index = np.argmax(new_solution[:, j])
                if new_solution[col_max_index][j] >= diff_value_2:
                    new_solution[col_max_index][j] -= diff_value_2
                    arr_sum_col -= diff_value_2
                    continue
            pos_num_2 = (new_solution[:, j] > 0).sum()
            num_col = np.round(diff_value_2 / pos_num_2, decimals=5)
            # 将该列的每个数都减去一个值，直到符合要求
            new_solution[:, j] -= num_col
            new_solution[:, j] = new_solution[:, j] * (new_solution[:, j] >= 0)
            arr_sum_col = new_solution[:, j].sum()
    return np.round(new_solution, decimals=5)

# 对粒子的位置进行检查
def CheckSolution(solution, overCld, underCld, cloudlets, flag=False):
    """
    :param solution: 粒子的解
    :param overCld: 过载微云序号集合
    :param underCld: 不过载微云序号集合
    :param cloudlets: 微云集合
    """
    len_Vs = len(overCld)
    len_Vt = len(underCld)
    # 首先检查解当中是否有负数存在，存在则置为0，保证解中的每个数都>=0
    new_solution = np.maximum(solution, 0)
    # 接下来对解的每行进行检查，防止每行值之和超过过载微云i的任务到达率
    for i in range(len_Vs):
        # 微云i的迁出的任务流之和
        cloud_arr_rate = cloudlets[overCld[i]].arrivalRate
        arr_sum_row = new_solution[i, :].sum()
        while arr_sum_row > cloud_arr_rate:
            # 正数的个数,因为True相当于1，可以直接求和代表个数
            pos_num_1 = (new_solution[i, :] > 0).sum()
            # 将该行的每个数都减去一个值，直到符合要求（⭐⭐⭐注意除数要为>0的个数，不然会出现循环，因为对为0的数减少，之后又置为0，该数就应该不要算进去）
            diff_value_1 = arr_sum_row - cloud_arr_rate
            num_row = np.round(diff_value_1 / pos_num_1, decimals=5)
            if num_row < 1e-3:
                row_max_index = np.argmax(new_solution[i, :])
                new_solution[i][row_max_index] -= diff_value_1
                arr_sum_row = new_solution[i, :].sum()
                continue
            new_solution[i, :] -= num_row
            # 相减出现负值的话，将负值置为0
            new_solution[i, :] = new_solution[i, :] * (new_solution[i, :] >= 0)
            arr_sum_row = new_solution[i, :].sum()
    # 接下来对解的每列进行检查，防止每列值之和超过不过载微云j的总任务接受率(⭐⭐⭐要考虑不过载微云本身的任务到达率)
    for j in range(len_Vt):
        arg = cloudlets[underCld[j]].serverNum * cloudlets[underCld[j]].serverRate - \
              cloudlets[underCld[j]].arrivalRate
        arg = arg - 1e-3
        arr_sum_col = new_solution[:, j].sum()
        # 注意：这里的条件是sum>=arg，与上面条件不一样，要有所修改
        while arr_sum_col > arg:
            pos_num_2 = (new_solution[:, j] > 0).sum()
            # 将该列的每个数都减去一个值，直到符合要求
            diff_value_2 = arr_sum_col - arg
            num_col = np.round(diff_value_2 / pos_num_2, decimals=5)
            if num_col < 1e-3:
                col_max_index = np.argmax(new_solution[:, j])
                new_solution[col_max_index][j] -= diff_value_2
                arr_sum_col = new_solution[:, j].sum()
                continue
            new_solution[:, j] -= num_col
            new_solution[:, j] = new_solution[:, j] * (new_solution[:, j] >= 0)
            arr_sum_col = new_solution[:, j].sum()
    return np.round(new_solution, decimals=5)
