# ------------------------------
# GenerateCloudlets.py: 用于生成微云集合及其数据
# -----------------------------
from PGCloudlet import Cloudlet, Cloudlets
import pickle
import numpy as np
# 微云个数
K = 20
# 初始化微云集合
cloudlet = Cloudlet()
cloudlets = Cloudlets(cloudlet, K)
cloudlets.initialize()
# 生成参数
ser_num = cloudlets.initServerNum()
ser_rate = cloudlets.initServerRate()
arr_rate = cloudlets.initArrivalRate()
delay_matrix = cloudlets.initC()
print('服务器数目: ', ser_num)
print('服务器速率: ', ser_rate)
print('任务到达率: ', arr_rate)
print('延时矩阵: ', delay_matrix)
# 将数据写入pickle文件
pickle.dump(ser_num, open('serNum.pkl', 'wb'))
pickle.dump(ser_rate, open('serRate.pkl', 'wb'))
pickle.dump(arr_rate, open('arrRate.pkl', 'wb'))
pickle.dump(delay_matrix, open('delayMat.pkl', 'wb'))
# np.savetxt('serNum.txt', ser_num, fmt='%0.5f')
# np.savetxt('serRate.txt', ser_rate, fmt='%0.5f')
# np.savetxt('arrRate.txt', arr_rate, fmt='%0.5f')
# np.savetxt('delayMat.txt', delay_matrix, fmt='%0.5f')
