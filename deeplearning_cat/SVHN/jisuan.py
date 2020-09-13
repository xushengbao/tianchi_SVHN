# -*- coding:utf-8 -*-
#多次计算得到kmeans聚类后的均值anchor

a = [[15, 65], [22, 108], [25, 176], [33, 139], [34, 209], [43, 179], [43, 242], [53, 224], [67, 253]]
b = [[14.34977578475337, 62.4390243902439], [20.392156862745097, 104.55445544554456], [25.599999999999987, 186.66666666666669], [29.57983193277311, 125.81196581196582], [35.55555555555557, 166.95652173913047], [37.647058823529406, 227.0967741935484], [46.153846153846146, 188.23529411764702], [51.20000000000001, 237.4193548387097], [66.20689655172413, 249.7560975609756]]
import numpy as np
f = "D:/project/python3.6/Code/gongju/kmeans-anchor-boxes-master/anchor.txt"
f = open(f)  # 如果你的x.txt文件不在python的路径下,那么必须用绝对路径
l1 = f.readlines()  # 这时候l1的结果是一个list,每个元素是文件的每一行,包括转行符号'/n'
l1 = [x.split(' ') for x in l1]
l1 = [[x[0], x[1], x[2], x[3], x[4], x[5],x[6], x[7], x[8], x[9], x[10], x[11],x[12], x[13], x[14], x[15], x[16], x[17].replace('/n', '')] for x in l1]
# print(l1)
l2 = [[float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]),float(x[6]), float(x[7]), float(x[8]), float(x[9]), float(x[10]), float(x[11]),float(x[12]), float(x[13]), float(x[14]), float(x[15]), float(x[16]), float(x[17])] for x in l1]
# print(l2)
l2= np.array(l2)
# print(l2)
out = np.zeros(18)
print(out)
out = np.array(out)
for x in l2:
    print(x)
    out += 0.1*x
print(out)
out = out.tolist()
#out = [[int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5]),int(x[6]), int(x[7]), int(x[8]), int(x[9]), int(x[10]), int(x[11]),int(x[12]), int(x[13]), int(x[14]), int(x[15]), int(x[16]), int(x[17])] for x in out]
out_new = [int(x) for x in out]
print(out_new)