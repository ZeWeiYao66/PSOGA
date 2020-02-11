import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from PGCloudlet import Cloudlet, Cloudlets
from PGComponent import Individual, Population
from PGOperator import Mutation
from PSOGA import Psoga
import time
import pickle

'''Step1：参数设置'''
K = 20  # 微云个数
N = 1000  # 迭代数目
Pnum = 100  # 种群数目
Rmut = 0.02  # 变异概率
w = 0.8  # 惯性权重
c1 = 2  # 学习因子
c2 = 2
# serNum = [1, 1, 2, 4, 3, 1, 5, 2, 5, 2, 2, 2, 1, 4, 2, 4, 6, 4, 2, 4, 4, 1, 4, 1, 1, 5, 2, 2, 1, 3, 7, 6, 3, 6, 3, 1, 2, 6, 2, 3]
# serRate = [4.64643, 7.62997, 2.87965, 4.24972, 1.83259, 6.07484, 2.67371, 3.46559, 2.50807, 5.59598, 2.7466, 7.89132, 6.86592, 3.77122, 4.57702, 8.41931, 9.06884, 7.52743, 5.33381, 10.2629, 5.92391, 3.20428, 2.96543, 7.00117, 6.97936, 5.36808, 4.68545, 3.69228, 5.45013, 4.7979, 5.34681, 5.67619, 5.77098, 4.83012, 5.12163, 5.23182,6.92917, 2.97366, 5.84462, 4.7285]
# arrRate = [3.44644, 4.67736, 4.92604, 9.63838, 4.1281, 4.01115, 9.15665, 4.61328, 10.99064, 9.50959, 2.90163, 8.65859,0.80629, 13.97776, 6.54769, 9.15579, 11.88867, 18.92704, 9.16532, 14.39091, 15.52477, 1.77773, 8.90902, 6.15746, 6.71168, 21.32335, 8.30922, 3.57061, 4.2884, 8.14829, 14.56103, 15.26725, 10.51691, 20.32832, 14.69924, 0.26997, 6.04588, 15.39505, 10.89264, 9.64849]
serNum = [2, 4, 6, 2, 3, 2, 2, 5, 2, 2, 2, 5, 6, 1, 3, 4, 3, 1, 2, 3]
serRate = [8.3381, 5.4488, 0.5407, 10.6447, 8.4327, 5.3003, 10.2055, 5.2503, 7.9187, 6.9716, 9.7556, 5.7564, 9.4682,
           6.4017, 2.7127, 3.8571, 9.1838, 2.0727, 2.4533, 3.777]
arrRate = [10.8156, 9.3921, 3.2423, 11.8673, 10.5512, 10.3321, 10.8762, 10.5384, 9.8906, 10.7433, 11.1472, 10.4119,
           10.6084, 6.3255, 7.4691, 8.8633, 10.2836, 2.0448, 4.8502, 9.616]
'''Step2：对每个微云i的服务率，服务器数量，任务到达率,网络延时进行初始化'''
cloudlet = Cloudlet()  # 单个微云
cloudlets = Cloudlets(cloudlet, K)  # 微云集合
cloudlets.initialize()  # 初始化微云集合
np.set_printoptions(threshold=1e6)
np.set_printoptions(suppress=True)
for i in range(K):
    cloudlets.cloudlets[i].serverNum = serNum[i]
    cloudlets.cloudlets[i].serverRate = serRate[i]
    cloudlets.cloudlets[i].arrivalRate = arrRate[i]

# with open(r"E:\\毕设\\文档\\a.txt", "rb") as f:
#     delayMat = pickle.load(f)
cloudlets.C = [
    [0, 0.2827, 0.1661, 0.2036, 0.1583, 0.2025, 0.2161, 0.2168, 0.1506, 0.2359, 0.2094, 0.2445, 0.2174, 0.1774, 0.1879,
     0.158, 0.1753, 0.2742, 0.2469, 0.212],
    [0.2827, 0, 0.2683, 0.1874, 0.2093, 0.2365, 0.1508, 0.2777, 0.227, 0.1572, 0.2073, 0.2055, 0.2525, 0.1785, 0.2979,
     0.2004, 0.2198, 0.1818, 0.1972, 0.1772],
    [0.1661, 0.2683, 0, 0.2183, 0.2131, 0.2386, 0.213, 0.1847, 0.2186, 0.2032, 0.2177, 0.1958, 0.16, 0.1976, 0.1571,
     0.2461, 0.2472, 0.1534, 0.2006, 0.2255],
    [0.2036, 0.1874, 0.2183, 0, 0.1688, 0.1841, 0.2194, 0.1821, 0.2265, 0.2282, 0.2278, 0.2257, 0.2057, 0.1908, 0.1759,
     0.2436, 0.2067, 0.1803, 0.1669, 0.2009],
    [0.1583, 0.2093, 0.2131, 0.1688, 0, 0.2122, 0.2663, 0.2529, 0.158, 0.2199, 0.2104, 0.2632, 0.2623, 0.2184, 0.1786,
     0.2258, 0.25, 0.1557, 0.1588, 0.1889],
    [0.2025, 0.2365, 0.2386, 0.1841, 0.2122, 0, 0.2807, 0.2455, 0.1815, 0.2792, 0.1552, 0.1871, 0.2319, 0.2166, 0.1685,
     0.1645, 0.2244, 0.2491, 0.1712, 0.2292],
    [0.2161, 0.1508, 0.213, 0.2194, 0.2663, 0.2807, 0, 0.1848, 0.1767, 0.2346, 0.2255, 0.1971, 0.1605, 0.1622, 0.1969,
     0.2049, 0.197, 0.1627, 0.1684, 0.23],
    [0.2168, 0.2777, 0.1847, 0.1821, 0.2529, 0.2455, 0.1848, 0, 0.2024, 0.2731, 0.2223, 0.2436, 0.2583, 0.1823, 0.2369,
     0.1666, 0.2089, 0.182, 0.193, 0.2913],
    [0.1506, 0.227, 0.2186, 0.2265, 0.158, 0.1815, 0.1767, 0.2024, 0, 0.1822, 0.1686, 0.183, 0.1567, 0.2527, 0.2393,
     0.2477, 0.1663, 0.2095, 0.1645, 0.2443],
    [0.2359, 0.1572, 0.2032, 0.2282, 0.2199, 0.2792, 0.2346, 0.2731, 0.1822, 0, 0.217, 0.2395, 0.2516, 0.2428, 0.2656,
     0.2738, 0.2663, 0.2684, 0.2356, 0.2065],
    [0.2094, 0.2073, 0.2177, 0.2278, 0.2104, 0.1552, 0.2255, 0.2223, 0.1686, 0.217, 0, 0.2549, 0.1534, 0.2103, 0.1741,
     0.1769, 0.2195, 0.1625, 0.1811, 0.2239],
    [0.2445, 0.2055, 0.1958, 0.2257, 0.2632, 0.1871, 0.1971, 0.2436, 0.183, 0.2395, 0.2549, 0, 0.2225, 0.2187, 0.192,
     0.2524, 0.2123, 0.2544, 0.1836, 0.1654],
    [0.2174, 0.2525, 0.16, 0.2057, 0.2623, 0.2319, 0.1605, 0.2583, 0.1567, 0.2516, 0.1534, 0.2225, 0, 0.1978, 0.2218,
     0.2202, 0.1574, 0.1652, 0.2229, 0.2027],
    [0.1774, 0.1785, 0.1976, 0.1908, 0.2184, 0.2166, 0.1622, 0.1823, 0.2527, 0.2428, 0.2103, 0.2187, 0.1978, 0, 0.2336,
     0.1567, 0.2066, 0.2009, 0.2339, 0.1975],
    [0.1879, 0.2979, 0.1571, 0.1759, 0.1786, 0.1685, 0.1969, 0.2369, 0.2393, 0.2656, 0.1741, 0.192, 0.2218, 0.2336, 0,
     0.1611, 0.1611, 0.2124, 0.1783, 0.1913],
    [0.158, 0.2004, 0.2461, 0.2436, 0.2258, 0.1645, 0.2049, 0.1666, 0.2477, 0.2738, 0.1769, 0.2524, 0.2202, 0.1567,
     0.1611, 0, 0.2371, 0.2569, 0.2165, 0.1928],
    [0.1753, 0.2198, 0.2472, 0.2067, 0.25, 0.2244, 0.197, 0.2089, 0.1663, 0.2663, 0.2195, 0.2123, 0.1574, 0.2066,
     0.1611, 0.2371, 0, 0.1658, 0.2945, 0.1696],
    [0.2742, 0.1818, 0.1534, 0.1803, 0.1557, 0.2491, 0.1627, 0.182, 0.2095, 0.2684, 0.1625, 0.2544, 0.1652, 0.2009,
     0.2124, 0.2569, 0.1658, 0, 0.2097, 0.286],
    [0.2469, 0.1972, 0.2006, 0.1669, 0.1588, 0.1712, 0.1684, 0.193, 0.1645, 0.2356, 0.1811, 0.1836, 0.2229, 0.2339,
     0.1783, 0.2165, 0.2945, 0.2097, 0, 0.1965],
    [0.212, 0.1772, 0.2255, 0.2009, 0.1889, 0.2292, 0.23, 0.2913, 0.2443, 0.2065, 0.2239, 0.1654, 0.2027, 0.1975,
     0.1913, 0.1928, 0.1696, 0.286, 0.1965, 0]
]
'''Step2：计算每个微云的本地任务响应时间'''
waitTimes = cloudlets.CalWaitTimes()
print('所有微云的本地任务响应时间: ', waitTimes)
print('其中最大的本地任务响应时间: ', max(waitTimes))
waitTimes_sorted = np.sort(waitTimes)  # 按照微云的本地任务响应时间进行排序
'''Step3：按照微云的本地任务响应时间，将K个微云划分成过载和不过载的微云.
          从第二个微云开始，到倒数第二个结束.'''
fitness = []  # 存放适应度值
result = []  # 存放每一次划分的结果
start_time = time.process_time()
for i in range(5, K - 5):
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
    '''Step4：初始化粒子的解（也即任务流g）,计算适应值，并进行迭代'''
    I = Individual()  # 单个粒子
    P = Population(I, cloudlets, OverCloudlet, UnderCloudlet, Pnum, w)  # 种群(粒子集合)
    # print(P.gbestFitness)
    M = Mutation(Rmut)  # 变异操作
    psoga = Psoga(P, M)
    bestResult, fit = psoga.run(N)  # 运行主算法，获得每一次划分的最优解及其适应度
    result.append(bestResult)
    fitness.append(fit)
end_time = time.process_time()
print('total time outside:', end_time - start_time)
'''Step5：从result中计算最优解'''
best_index = np.argmin(fitness)  # 获取最优结果的下标
print(fitness[best_index])  # 打印最优适应度值
print(result[best_index])  # 打印最优解

# 服务器数目： [1, 1, 2, 4, 3, 1, 5, 2, 5, 2, 2, 2, 1, 4, 2, 4, 6, 4, 2, 4, 4, 1, 4, 1, 1, 5, 2, 2, 1, 3, 7, 6, 3, 6, 3, 1,
# 2, 6, 2, 3] 服务率： [4.64643, 7.62997, 2.87965, 4.24972, 1.83259, 6.07484, 2.67371, 3.46559, 2.50807, 5.59598, 2.7466,
# 7.89132, 6.86592, 3.77122, 4.57702, 8.41931, 9.06884, 7.52743, 5.33381, 10.2629, 5.92391, 3.20428, 2.96543,
# 7.00117, 6.97936, 5.36808, 4.68545, 3.69228, 5.45013, 4.7979, 5.34681, 5.67619, 5.77098, 4.83012, 5.12163, 5.23182,
# 6.92917, 2.97366, 5.84462, 4.7285]

# 任务到达率： [3.44644, 4.67736, 4.92604, 9.63838, 4.1281, 4.01115, 9.15665, 4.61328, 10.99064, 9.50959, 2.90163, 8.65859,
# 0.80629, 13.97776, 6.54769, 9.15579, 11.88867, 18.92704, 9.16532, 14.39091, 15.52477, 1.77773, 8.90902, 6.15746,
# 6.71168, 21.32335, 8.30922, 3.57061, 4.2884, 8.14829, 14.56103, 15.26725, 10.51691, 20.32832, 14.69924, 0.26997,
# 6.04588, 15.39505, 10.89264, 9.64849]
