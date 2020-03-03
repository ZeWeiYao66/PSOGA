import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from PGCloudlet import Cloudlet, Cloudlets
from PGComponent import Individual, Population
from PGOperator import Mutation, CrossOver
from PSOGA import Psoga
import time

'''Step1：参数设置'''
K = 20  # 微云个数
N = 1000  # 迭代数目
Pnum = 100  # 种群数目
Rmut = 0.05  # 变异概率
w = 0.8  # 惯性权重
c1 = 2  # 学习因子
c2 = 2

'''Step2：对每个微云i的服务率，服务器数量，任务到达率,网络延时进行初始化'''
cloudlet = Cloudlet()  # 单个微云
cloudlets = Cloudlets(cloudlet, K)  # 微云集合
serNum, serRate, arrRate, delayMat = cloudlets.initialize()  # 初始化微云集合
print('服务器数目：', serNum.tolist())
print('服务率：', serRate.tolist())
print('任务到达率：', arrRate)
print('网络延时：', delayMat.tolist())

'''Step3：计算每个微云的本地任务响应时间'''
waitTimes = cloudlets.CalWaitTimes()
print('所有微云的本地任务响应时间: ', waitTimes)
print('其中最大的本地任务响应时间: ', max(waitTimes))
# 按照微云的本地任务响应时间进行排序
waitTimes_sorted = np.sort(waitTimes)

'''Step4：按照微云的本地任务响应时间，将K个微云划分成过载和不过载的微云'''
fitness = []  # 存放适应度值
result = []  # 存放每一次划分的结果
start_time = time.process_time()
# 根据测试得出的最优解主要分布的范围
new_k = int(K/2)
for i in range(new_k-3, new_k+3):
    OverCloudlet = []  # 过载微云的序号集合
    UnderCloudlet = []  # 不过载微云的序号集合
    Tp = waitTimes_sorted[i]  # 用作划分微云集合的微云p的本地任务响应时间
    # 将微云划分成两个集合
    for j in range(K):
        # 过载集合
        if waitTimes[j] > Tp:
            OverCloudlet.append(j)
        # 不过载集合
        else:
            UnderCloudlet.append(j)
    print('第 ' + str(i) + ' 次')
    print('过载微云集合:  ', OverCloudlet)
    print('不过载微云集合:', UnderCloudlet)

    '''Step5：初始化粒子的解（也即任务流g）,计算适应值，并进行迭代'''
    I = Individual()  # 单个粒子
    P = Population(I, cloudlets, OverCloudlet, UnderCloudlet, Pnum, w)  # 种群(粒子集合)
    # print(P.gbestFitness)
    M = Mutation(Rmut)  # 变异操作
    C = CrossOver()     # 交叉操作
    psoga = Psoga(P, M, C)
    bestResult, fit = psoga.run(N)  # 运行主算法，获得每一次划分的最优解及其适应度
    result.append(bestResult)
    fitness.append(fit)
end_time = time.process_time()
print('total time outside:', end_time - start_time)

'''Step6：从result中计算最优解'''
best_index = np.argmin(fitness)  # 获取最优结果的下标
print(fitness[best_index])  # 打印最优适应度值
print(result[best_index])  # 打印最优解
