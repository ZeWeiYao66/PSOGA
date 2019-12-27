# ------------------------------
# PGCloudlet.py: Cloudlet类、Cloudlets类，主要是与微云有关的
# ------------------------------

import numpy as np

'''
单个微云
'''
class Cloudlet:
    # 设定微云的参数
    def __init__(self):
        self.serverNum = None                 # 服务器数量n
        self.serverRate = None                # 服务率u
        self.arrivalRate = None               # 任务到达率
        self.waitTime = None                  # 任务等待时间T
        self.netDelay = None                  # 传入任务的总网络延时Tnet
        self.responseTime = None              # 任务响应时间D（D=T+Tnet）

    # 计算任务等待时间
    def CalWaitTime(self, cloudlet):
        pass

    # 计算对应微云的总网络延时
    def CalNetDelay(self, cloudlet):
        pass

    # 计算任务响应时间
    def CalResponseTime(self, cloudlet):
        pass


'''
微云集合
'''
class Cloudlets:
    # 设定微云集合
    def __init__(self, cloudlet, K=40):
        self.K = K                                  # 微云数目
        self.cloudlet = cloudlet                    # 微云模板
        self.cloudlets = None                       # 微云集合
        self.C = None                               # 接入点之间的网络延时C(i,j)

    # 初始化微云集合
    def initialize(self):
        pass

    # 初始化微云的服务器数量，服从泊松分布（均值为3）
    def initServerNum(self, cloudlets):
        pass

    # 初始化微云的服务率，服从正态分布N(5,2)>0
    def initServerRate(self, cloudlets):
        pass

    # 初始化微云的任务到达率，服从正态分布N(5,2)>0
    def initArrivalRate(self, cloudlets):
        pass

    # 初始化微云的网络延时C
    def initC(self):
        pass
