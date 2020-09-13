# -*- coding:utf-8 -*-
# 对yolo3检测结果的txt文件转为提交格式的csv
import pandas as pd

c = 'D:/data/SVHN/test/input/zhengli/tijiao_2.txt'
file = open(c, 'r')
data = file.readlines()
# data = "\n".join(data)
# print(data)
#data = data.replace('\n','')
df_submit = pd.read_csv('D:/data/SVHN/test/input/zhengli/mchar_sample_submit_A.csv')
df_submit['file_code'] = data
df_submit.to_csv('D:/data/SVHN/test/input/zhengli/submit.csv', index=None)
