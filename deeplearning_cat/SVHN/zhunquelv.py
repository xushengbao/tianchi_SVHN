#比较街景识别比赛中不同提交结果的相似度
# import pandas as pd
# df_submit = pd.read_csv('D:/data/SVHN/tijiao/submit3.csv')
# # df_submit['file_code'] = test_label_pred
# print(df_submit['file_code'])
# data = df_submit['file_code']
# #print(data.type)
# df_submit.to_csv('sub mit.csv', index=None)
# file = open('D:/data/SVHN/tijiao/submit3.txt', 'w')
# file.write(data)

import csv
with open('D:/data/SVHN/tijiao_CNN/submit5.csv','rt',encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    rows3 = [row for row in reader]
    #print(rows3[1][1])
    #print(rows)
#a = ['039816.png', '100']
with open('D:/data/SVHN/test/yolo3/submit.csv','rt',encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    #print(reader)
    rows4 = [row for row in reader]
    #print(rows4[1][1])
a = 0
for row3 in rows3:
    #print(row3)
    for row4 in rows4:
        #print(row4)
        if row3 == row4:
            a+=1

print(a)

