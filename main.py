import numpy as np
from PGCloudlet import Cloudlet, Cloudlets
from PGComponent import Individual, Population
from PGOperator import Mutation
from PSOGA import Psoga
import time
from Utils import save

'''Step1：参数设置'''
K = 20              # 微云个数
N = 1000            # 迭代数目
Pnum = 100          # 种群数目
Rmut = 0.05         # 变异概率
w = 0.8             # 惯性权重
c1 = 2              # 学习因子
c2 = 2
# 读取生成的微云参数(参数设置成list可以节省时间，使用numpy.ndarray会比较长⭐⭐⭐)
serNum = np.loadtxt('serNum.txt', dtype=np.int).tolist()  # 服务器数目
serRate = np.loadtxt('serRate.txt').tolist()              # 服务器速率
arrRate = np.loadtxt('arrRate.txt').tolist()              # 任务到达率
delayMatrix = np.loadtxt('delayMat.txt').tolist()         # 网络延时
# with open("serNum.pkl", "rb") as f:
#     serNum = pickle.load(f).tolist()
# with open("serRate.pkl", "rb") as f:
#     serRate = pickle.load(f).tolist()
# with open("arrRate.pkl", "rb") as f:
#     arrRate = pickle.load(f)
# with open("delayMat.pkl", "rb") as f:
#     delayMatrix = pickle.load(f).tolist()
'''Step2：对每个微云i的服务率，服务器数量，任务到达率,网络延时进行初始化'''
time1 = time.process_time()
cloudlet = Cloudlet()               # 单个微云
cloudlets = Cloudlets(cloudlet, K)  # 微云集合
cloudlets.initialize()              # 初始化微云集合
print('服务器数目：', serNum)
print('服务率：', serRate)
print('任务到达率：', arrRate)
print('网络延时：', delayMatrix)
# 对初始化后的微云集合进行赋值
for i in range(K):
    cloudlets.cloudlets[i].serverNum = serNum[i]
    cloudlets.cloudlets[i].serverRate = serRate[i]
    cloudlets.cloudlets[i].arrivalRate = arrRate[i]
cloudlets.C = delayMatrix

'''Step3：计算每个微云的本地任务响应时间'''
waitTimes = cloudlets.CalWaitTimes()
print('所有微云的本地任务响应时间: ', waitTimes)
print('其中最大的本地任务响应时间: ', max(waitTimes))
# 按照微云的本地任务响应时间进行排序
waitTimes_sorted = np.sort(waitTimes)
time2 = time.process_time()
print('初始化微云时间: ', time2 - time1)

times = []          # 存储运行10次所花费的时间
bestFitness = []    # 存储运行10次所求得的最优适应度值
# 运行10次
for _ in range(10):
    '''Step4：按照微云的本地任务响应时间，将K个微云划分成过载和不过载的微云'''
    fitness = []  # 存放一次运行对应所有划分的最优适应度值
    result = []   # 存放依次运行所对应的所有划分的结果
    OverCloudlet = []  # 过载微云的序号集合
    UnderCloudlet = []  # 不过载微云的序号集合
    start_time = time.time()
    new_k = int(K / 2)
    Tp = waitTimes_sorted[new_k - 4]  # 用于划分微云集合的微云p的本地任务响应时间
    # 先将第new_k-4个微云作为划分标准，将微云划分成两个集合
    for j in range(K):
        # 过载集合
        if waitTimes[j] > Tp:
            OverCloudlet.append(j)
        # 不过载集合
        else:
            UnderCloudlet.append(j)
    # 根据测试得出的最优解主要分布的范围
    for i in range(new_k - 3, new_k + 4):
        Tp = waitTimes_sorted[i]  # 用作划分微云集合的微云p的本地任务响应时间
        index_tp = waitTimes.index(Tp)  # 从waitTimes数组找到对应的响应时间下标
        OverCloudlet.remove(index_tp)  # 更新过载微云集合、不过载微云集合
        UnderCloudlet.append(index_tp)
        UnderCloudlet.sort()  # 对不过载微云进行排序，保证是升序排列
        print('第 ' + str(i) + ' 次')
        print('过载微云集合:  ', OverCloudlet)
        print('不过载微云集合:', UnderCloudlet)
        '''Step5：初始化粒子的解（也即任务流g）,计算适应值，并进行迭代'''
        I = Individual()  # 单个粒子
        P = Population(I, cloudlets, OverCloudlet, UnderCloudlet, arrRate, Pnum, w)  # 种群(粒子集合)
        M = Mutation(Rmut)  # 变异操作
        psoga = Psoga(P, M)
        bestResult, fit = psoga.run(N)  # 运行主算法，获得每一次划分的最优解及其适应度
        result.append(bestResult)
        fitness.append(fit)
    end_time = time.time()
    print('total time outside:', end_time - start_time)
    '''Step6：从result中计算最优解'''
    best_index = np.argmin(fitness)  # 获取最优结果的下标
    times.append(end_time - start_time)      # 将该次运行的运行时间保存起来
    bestFitness.append(fitness[best_index])  # 将该次运行的最优结果保存起来
    print(fitness[best_index])  # 打印最优适应度值
    print(result[best_index])  # 打印最优解
print('时间：', times)
print("最优适应度值：", bestFitness)
a1 = [str(i) for i in bestFitness]
a2 = [str(j) for j in times]
save(r'E:\毕设\文档\fitness.txt', a1)
save(r'E:\毕设\文档\times.txt', a2)

