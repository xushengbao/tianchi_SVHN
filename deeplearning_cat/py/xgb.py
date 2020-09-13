#处理街景数据中，训练与验证集合并后的命名
# from __future__ import print_function
# import os, sys, zipfile
# import json
#
#
# def convert(size, box):
#     dw = 1. / (size[0])
#     dh = 1. / (size[1])
#     x = box[0] + box[2] / 2.0
#     y = box[1] + box[3] / 2.0
#     w = box[2]
#     h = box[3]
#
#     x = x * dw
#     w = w * dw
#     y = y * dh
#     h = h * dh
#     return (x, y, w, h)
#
#
# json_file = 'D:/data/SVHN/train/train_annos.json'  # # Object Instance 类型的标注
#
# data = json.load(open(json_file, 'r'))
#
# ana_txt_save_path = "D:/data/SVHN/train/test"  # 保存的路径
# if not os.path.exists(ana_txt_save_path):
#     os.makedirs(ana_txt_save_path)
#
# for img in data['images']:
#     # print(img["file_name"])
#     filename = img["filename"]
#     img_width = img["width"]
#     img_height = img["height"]
#     # print(img["height"])
#     # print(img["width"])
#     img_id = img["id"]
#     ana_txt_name = filename.split(".")[0] + ".txt"  # 对应的txt名字，与jpg一致
#     print(ana_txt_name)
#     f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'w')
#     for ann in data['annotations']:
#         if ann['image_id'] == img_id:
#             # annotation.append(ann)
#             # print(ann["category_id"], ann["bbox"])
#             box = convert((img_width, img_height), ann["bbox"])
#             f_txt.write("%s %s %s %s %s\n" % (ann["category_id"], box[0], box[1], box[2], box[3]))
#     f_txt.close()


import json

annotation_path = "D:/data/SVHN/val/train_val.json"
data = json.load(open(annotation_path, 'r'))


new_data = {}

c = 29999
for i in range(10000,20000):
    c += 1
    a1 = (i // 1000)%10
    a2 = (i // 100)%10
    a3 = (i // 10)%10
    a4 = i % 10
    #print(a1, a2, a3, a4)
    a = '00' + str(a1) + str(a2) + str(a3) + str(a4) + '.png'
    # print(a)
    b = '0' + str(c) + '.png'
    #print(data[a])
    new_data[b] = data[a]
print(new_data)

json_str = json.dumps(new_data)
with open('D:/data/SVHN/val/train_valdata.json', 'w') as json_file:
    json_file.write(json_str)
    # with open(annotation_path, encoding="utf-8") as f:
    #     # 读取所有行 每行会是一个字符串
    #     data = f.readlines()
    # # 将josn字符串转化为dict字典

#         if a in data:
#             data.replace(a, b)
#         c += 1
#
# # print(data)
