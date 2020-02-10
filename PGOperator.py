# ------------------------------
# PGOperator.py: Mutation类
# ------------------------------
import numpy as np
import Utils


class Mutation:
    # 初始化
    def __init__(self, rate):
        """
        :param rate:变异概率
        """
        self.rate = rate

    # 变异操作（随机选取行和列，对这些行和列的值进行重新初始化）
    def mutate(self, individual, overCld, underCld, cloudlets):
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
        np.random.shuffle(row_rand_array)  # 对行下标进行重新排列
        np.random.shuffle(col_rand_array)  # 对列下标进行重新排列
        # 相当于对|len1|×|len2|的矩阵重新初始化
        # len1 = Utils.calculate_len(len_Vs)  # 随机选取其中的一些行,len1为随机选取的个数
        len1 = np.random.randint(2, 20) if (len_Vs > 20) else (np.random.randint(2, len_Vs) if (len_Vs > 2) else len_Vs)
        row_rand_array = row_rand_array[:len1]
        # len2 = Utils.calculate_len(len_Vt)  # 随机选取其中的一些列,len2为随机选取的个数
        len2 = np.random.randint(2, 20) if (len_Vt > 20) else (np.random.randint(2, len_Vt) if (len_Vt > 2) else len_Vt)
        col_rand_array = col_rand_array[:len2]
        # 重新初始化
        for i in range(len1):
            oIndex = row_rand_array[i]  # 选取解的第oIndex行
            arriveRate = cloudlets[overCld[oIndex]].arrivalRate  # 取对应行的过载微云的任务到达率
            for j in range(len2):
                uIndex = col_rand_array[j]
                # 重新初始化
                individual[oIndex][uIndex] = np.round(np.random.uniform(0, arriveRate), decimals=5)
