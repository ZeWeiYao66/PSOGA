import time


# 粒子群遗传算法
class Psoga:
    def __init__(self, population, mutation, Wmin=0.3, Wmax=0.8):
        """
        :param population: 粒子群
        :param mutation: 变异操作
        :param Wmin: 惯性权重下限
        :param Wmax: 惯性权重上限
        """
        self.population = population  # 种群
        self.mutation = mutation  # 变异操作
        self.Wmin = Wmin  # 惯性权重取值范围
        self.Wmax = Wmax

    def run(self, gen=100):
        """
        :param gen: 种群迭代数
        :return: 全局极值
        """
        # Step1：初始化粒子群的解、速度，同时设置个体极值与全局极值
        start_time1 = time.time()
        self.population.initialize()
        end_time1 = time.time()
        print('initialize time: ', end_time1 - start_time1)
        # Step2：进行种群的迭代
        start_time2 = time.time()
        size = self.population.size
        overCld = self.population.overCld
        underCld = self.population.underCld
        cloudlets = self.population.cloudlets.cloudlets
        # 进行迭代
        for n in range(1, gen + 1):
            # Step3：对所有粒子进行速度、位置的更新，然后对全部粒子进行变异操作，在更新个体极值与全局极值
            # 计算惯性权重
            self.population.w = self.Wmax - n * (self.Wmax - self.Wmin) / gen
            # 1).更新粒子的位置、速度
            for index in range(size):
                self.population.update_position(index)
            # 2).进行变异操作
            self.mutation.mutate(self.population, overCld, underCld, cloudlets)
            # 3).计算粒子的适应度值
            self.population.CalculateAllFit()
            # 4).更新个体极值(全部粒子的速度和位置更新完之后再去计算)
            self.population.update_pbest()
            # 5).更新全局极值
            self.population.update_gbest()
        end_time2 = time.time()
        print('iterate time: ', end_time2 - start_time2)
        # Step4：结束迭代，返回全局极值
        print('最优解:  ', self.population.gbest)
        print('适应度值:', self.population.gbestFitness)
        print('响应时间:', self.population.responTime)
        return self.population.gbest, self.population.gbestFitness
