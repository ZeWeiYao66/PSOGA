# ------------------------------
# PGCloudlet.py: Cloudlet类、Cloudlets类，主要是与微云有关的
# ------------------------------
# np.around(a,decimals=)可以指定小数点后保留几位
import numpy as np
import scipy.stats as stats
import Utils

'''
单个微云
'''


class Cloudlet:
    # 设定微云的参数
    def __init__(self):
        self.serverNum = None  # 服务器数量n
        self.serverRate = None  # 服务率u
        self.arrivalRate = None  # 任务到达率
        self.waitTime = None  # 任务等待时间T
        self.netDelay = None  # 传入任务的总网络延时Tnet
        self.responseTime = None  # 任务响应时间D（D=T+Tnet）

    # 计算微云的任务等待时间
    def CalWaitTime(self):
        # 计算ErlangC的值
        erlangC = Utils.ErlangC(self.serverNum, self.arrivalRate / self.serverRate)
        T = erlangC / (self.serverNum * self.serverRate - self.arrivalRate) + 1 / self.serverRate
        self.waitTime = np.around(T, decimals=5)
        return self.waitTime

    # 计算对应微云的总网络延时
    def CalNetDelay(self):
        pass



'''
微云集合
'''


class Cloudlets:
    # 设定微云集合
    def __init__(self, cloudlet, K=40):
        self.K = K  # 微云数目
        self.cloudlet = cloudlet  # 微云模板
        self.cloudlets = None  # 微云集合(不要对其位置进行交换)
        self.C = None  # 接入点之间的网络延时C(i,j)

    # 初始化微云集合，微云服务器数量，服务率，任务到达率以及网络延时
    def initialize(self):
        CldClass = self.cloudlet.__class__
        self.cloudlets = np.array([CldClass() for i in range(self.K)], dtype=CldClass)
        ser_num = self.initServerNum()
        ser_rate = self.initServerRate()
        arr_rate = self.initArrivalRate()
        delay_matrix = self.initC()
        return ser_num, ser_rate, arr_rate, delay_matrix

    # 初始化微云的服务器数量，服从泊松分布（均值为3）
    def initServerNum(self):
        # 使用numpy的poisson实现泊松分布，返回K个微云的服务器数
        """Problem1: 该项可能会产生0，会使得初始化任务到达率出错。
                     如果出现0或者均值不为3，我们就重新生成"""
        while True:
            serNum = np.random.poisson(3, self.K)
            if (0 in serNum) or (np.sum(serNum) != 3 * self.K):
                continue
            else:
                break
        # 对微云集合进行赋值
        for i in range(self.K):
            self.cloudlets[i].serverNum = serNum[i]
        return serNum

    # 初始化微云的服务率，服从正态分布N(5,2)>0
    def initServerRate(self):
        # 指定上下限，均值和方差
        """这里把下限设置为0.25，是为了避免下面计算任务到达率的upper出现负值"""
        lower = 0.25
        upper = np.inf
        mu = 5
        sigma = 2
        # 对正态分布指定区间上下限，并对样本进行截取，并对提取出的样本进行精度设置
        serRate = np.around(stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma
                                                , loc=mu, scale=sigma, size=self.K), decimals=5)
        # 对微云集合进行赋值
        for i in range(self.K):
            self.cloudlets[i].serverRate = serRate[i]
        return serRate

    # 初始化微云的任务到达率，服从正态分布0<N(15,6)<u*n-0.25
    def initArrivalRate(self):
        arriveRate = []
        lower = 0
        mu = 15
        sigma = 6
        for i in range(self.K):
            # 计算upper
            upper = self.cloudlets[i].serverNum * self.cloudlets[i].serverRate - 0.25
            # 将结果加入arriveRate列表中
            arriveRate.append(stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma))
            self.cloudlets[i].arrivalRate = np.around(arriveRate[i], decimals=5)
        return np.around(arriveRate, decimals=5)

    # 初始化微云的网络延时C，0.1<=N(0.15,0.05)<=0.2(这里的延时矩阵C为对角线元素全为0的对称矩阵)
    def initC(self):
        lower = 0.1
        upper = 0.2
        mu = 0.15
        sigma = 0.05
        matrix = np.around(stats.truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma
                                               , size=(self.K, self.K)), decimals=5)
        # 保留矩阵上三角部分
        matrix = np.triu(matrix)
        # 将上三角拷贝到下三角部分
        matrix += matrix.T
        # 将对角线元素置为0
        for i in range(self.K):
            matrix[i][i] = 0
        self.C = matrix
        return self.C

    # 计算所有微云的任务等待时间
    def CalWaitTimes(self):
        waitTimes = []
        for i in range(self.K):
            waitTimes.append(self.cloudlets[i].CalWaitTime())
        return waitTimes


if __name__ == '__main__':
    cloudlet = Cloudlet()
    C = Cloudlets(cloudlet, 10)
    C.initialize()
    C.CalWaitTimes()
