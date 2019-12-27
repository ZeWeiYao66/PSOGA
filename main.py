import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from PGCloudlet import Cloudlet, Cloudlets
from PGComponent import Individual, Population
from PGOperator import Mutation
from PSOGA import Psoga

'''参数设置'''
K = 40              # 微云个数
N = 1000            # 迭代数目
Pnum = 100          # 种群数目
Rmut = 0.02         # 变异概率
w = 0.8             # 惯性权重
c1 = 2              # 学习因子
c2 = 2

'''Step1：对每个微云i的服务率，服务器数量，任务到达率,网络延时进行初始化'''
cloudlet = Cloudlet()
cloudlets = Cloudlets(cloudlet, K)
cloudlets.initialize()
cloudlets.initServerNum()
cloudlets.initServerRate()
cloudlets.initArrivalRate()
cloudlets.initC()
'''Step2：计算每个微云的本地任务响应时间'''
for i in range(K):
    cloudlets.cloudlets[i].CalWaitTime()
'''Step3：按照微云的本地任务响应时间，将K个微云划分成过载和不过载的微云.
          从第二个微云开始，到倒数第二个结束.'''
# 存放每次划分的结果
result = []
for i in range(1, K-1):
    # 将微云划分成两个集合
    # .......
    pass
    '''Step4：初始化粒子的解（也即任务流g）,计算适应值，并进行迭代'''
    I = Individual()
    P = Population(I, Pnum, w, c1, c2)
    M = Mutation(Rmut)
    psoga = Psoga(P, M)
    result.append(psoga.run(N))