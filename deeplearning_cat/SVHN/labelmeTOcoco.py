#把labelme数据转为coco格式的数据
import json
import os
import cv2
from tqdm import tqdm
dataset={"info": [],
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": []
        }
classes = ['0','1','2','3','4','5','6','7','8','9']
for i,cls in enumerate(classes,0):
    dataset['categories'].append({'id':i,'name':cls,'supercategory':'number'})
print(dataset)
# 读取所有训练集图片的名称
indexes = [f for f in os.listdir('D:/data/SVHN/train_val/train_val_img')]
print(len(indexes))

import pandas as pd
anno_file = 'D:/data/SVHN/train_val/train_val.json'
with open(anno_file, "r") as f_json:
    all_anno_pd = pd.read_json(f_json)
print(len(all_anno_pd))
print(all_anno_pd.iloc[0])

# for li,anno in enumerate(all_ann_pd):
# # print(all_anno_pd.index)
# print(all_anno_pd.loc['000000.png'])
# # print(all_anno_pd.loc['000000.png'][1][1])
# print(len(all_anno_pd.loc['000000.png'][1]))
# print(all_anno_pd.loc['000000.png'][1][0])
# for li,anno in enumerate(all_anno_pd.index):
#     print(li,anno)

all_anno_pd = all_anno_pd.T
all_anno_pd.to_csv('D:/data/SVHN/train_val/val_anno.txt',header = None,sep = '\t',index = True)
file = open("D:/data/SVHN/train_val/val_anno.txt","r")
list = file.readlines()

print(all_anno_pd.shape)

import json

with open('D:/data/SVHN/train_val/val_anno.txt', 'r') as f:
    annos = f.readlines()
# print(len(annos))
print(annos[0])

for li, anno in enumerate(annos):
    parts = anno.strip().split('\t')
    print(parts)
    label = json.loads(parts[2])
    print(label[0])

for k,index in tqdm(enumerate(indexes)):
    im = cv2.imread('D:/data/SVHN/train_val/train_val_img/'+index)
    height,width,channel = im.shape
    dataset['images'].append({
            'filename':index,
            'id':k,
            'width':width,
            'height':height
        })
count = 0
for li, anno in tqdm(enumerate(annos)):
    parts = anno.strip().split('\t')
    height = json.loads(parts[1])
    label = json.loads(parts[2])
    left = json.loads(parts[3])
    top = json.loads(parts[4])
    width = json.loads(parts[5])
    for k in range(len(label)):
        cls_id = label[k]
        x1 = float(left[k])
        y1 = float(top[k])
        x2 = float(width[k])
        y2 = float(height[k])
        count = count + 1
        dataset['annotations'].append({
            'area': x2 * y2,
            'bbox': [x1, y1, x2, y2],
            'category_id': int(cls_id),
            'id': count,
            'image_id': li,
            'iscrowd': 0,
            # mask, 矩形是从左上角点按顺时针的四个顶点
            'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
        })

#     for li,anno_imagename in enumerate(all_anno_pd.index):
#         for k in range(len(all_anno_pd.loc[anno_imagename][1])):
#             if anno_imagename == index:
#                 cls_id = all_anno_pd.loc[anno_imagename][1][k]
#                 x1 = float(all_anno_pd.loc[anno_imagename][2][k])
#                 y1 = float(all_anno_pd.loc[anno_imagename][3][k])
#                 x2 = float(all_anno_pd.loc[anno_imagename][4][k])
#                 y2 = float(all_anno_pd.loc[anno_imagename][0][k])
#                 dataset['annotations'].append({
#                     'area': x2 * y2,
#                     'bbox': [x1, y1, x2, y2],
#                     'category_id': int(cls_id),
#                     'id': li,
#                     'image_id': k,
#                     'iscrowd': 0,
#                     # mask, 矩形是从左上角点按顺时针的四个顶点
#                     'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
#               })
folder = 'D:/data/SVHN/train_val/annotations'
if not os.path.exists(folder):
    os.makedirs(folder)
json_name = 'D:/data/SVHN/train_val/annotations/{}.json'.format('train+val_annos')
with open(json_name, 'w') as f:
    json.dump(dataset, f)
print('over!!')


