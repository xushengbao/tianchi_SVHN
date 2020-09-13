# -*- coding:utf-8 -*-
# 统计训练数据集边框的left间隔距离
import json

train_json = json.load(open('D:/data/SVHN/train/mchar_train.json'))
# train_label = [train_json[x]['left'] for x in train_json if len(train_json[x]['label']) == 2]
# train_label1 = [train_json[x]['left'][0] for x in train_json if len(train_json[x]['label']) == 2]
# train_label2 = [train_json[x]['left'][1] for x in train_json if len(train_json[x]['label']) == 2]
# print(train_label)
# print(train_label1)
# print(train_label2)
#
#
# def list_sub(a,b):
#     c = []
#     for i in range(len(a)):
#         c.append(a[i]-b[i])
#     return c
# res = 0
# res1 =0
# res2 = 0
# res3 = 0
# sub = list_sub(train_label2,train_label1)
# print(sub)
# for i in range(len(train_label)):
#     if sub[i] == 1:
#         res += 1
#
#     if sub[i] == 2:
#         res1 += 1
#
#     if sub[i] == 3:
#         res2 += 1
#
#     if sub[i] == 4:
#         res3 += 1
#
# print(res)
# print(res1)
# print(res2)
# print(res3)

train_label = [train_json[x]['width'] for x in train_json]
print(len(train_label))
i = 0
for x in train_label:
    for y in x:
        if y <= 4:
            i += 1
print(i)



