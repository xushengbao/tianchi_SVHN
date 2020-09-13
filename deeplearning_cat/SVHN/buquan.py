#补全yolo3未检测出来的结果，用定长或者其他比较好的模型，进行结果的赋值
import pandas as pd
#
# df_submit1 = pd.read_csv('D:/data/SVHN/test/input/zhengli/test1.csv')
# #df_submit1 = pd.read_csv('D:/data/SVHN/tijiao/submit5.csv')
# df_submit2 = pd.read_csv('D:/data/SVHN/test/input/zhengli/test2.csv')
# #df_submit2 = pd.read_csv('D:/data/SVHN/test/input/zhengli/submit.csv')
# print(df_submit2['file_code'][1])
# for i in range(9):
#    if df_submit2['file_code'][i] >= 0:
#         df_submit2['file_code'][i] = df_submit1['file_code'][i]
# print(df_submit1['file_code'])
# print(df_submit2['file_code'])
# #df_submit.to_csv('D:/data/SVHN/test/input/zhengli/submit.csv', index=None)



import csv
csvfile1 = open('D:/data/SVHN/test/yolo3/yolov3_spp_512/thresh0.045/buquan.csv','rt',encoding="utf-8") #补全的模板
reader1 = csv.reader(csvfile1)
# print(reader1)
rows1 = [row for row in reader1]
print(rows1)
# with open('D:/data/SVHN/tijiao/submit5.csv','rt',encoding="utf-8") as csvfile:
#     reader = csv.reader(csvfile)
#     rows3 = [row for row in reader]
#     print(rows3[1][1])
    #print(rows)
#a = ['039816.png', '100']
csvfile2 = open('D:/data/SVHN/test/yolo3/submit.csv','rt',encoding="utf-8") #需要被补全的
reader2 = csv.reader(csvfile2)
rows2 = [row for row in reader2]
print(rows2)

data = []
for i in range(1,40001):
    if rows2[i][1] == '':
        rows2[i][1] = rows1[i][1]

    data.append(''.join(rows2[i][1]))
print(data)

# csvfile3 = open('D:/data/SVHN/test/input/zhengli/test3.csv','rt',encoding="utf-8")
df_submit = pd.read_csv('D:/data/SVHN/test/input0/zhengli/mchar_sample_submit_A.csv')
#df_submit = pd.read_csv('D:/data/SVHN/test/input/zhengli/test1.csv')
df_submit['file_code'] = data
df_submit.to_csv('D:/data/SVHN/test/yolo3/buquan.csv', index=None)   #存储补全的
print(rows1)
print(rows2)
