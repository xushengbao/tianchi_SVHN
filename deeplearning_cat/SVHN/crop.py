#剪裁训练和验证数据集
import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
train_json = json.load(open('D:/data/SVHN/val/mchar_val.json'))   #需要裁剪的图片的标签

# 数据标注处理
def parse_json(d):
   arr = np.array([
       d['top'], d['height'], d['left'], d['width'], d['label']
   ])
   arr = arr.astype(int)
   return arr


def list_sum(a, b):
   c = []
   for i in range(len(a)):
      c.append(a[i] + b[i])
   return c

# def hebing(d):
#
#    return arr

imagedir = 'D:/data/SVHN/val/mchar_val' #需要裁剪的图片路径
for item in os.listdir(imagedir):
    if item.endswith('.png'):
       #print(item)
       new_dict = {}
       image = imagedir + '/' + item
       img = cv2.imread(image)
       # biaoqian = {'height': [62, 63, 69], 'label': [6, 3, 9], 'left': [6, 43, 74], 'top': [16, 22, 21], 'width': [41, 30, 41]}
       #biaoqian = {'height': [69], 'label': [1], 'left': [6], 'top': [16], 'width': [112]}
       #arr = parse_json(biaoqian)
       #print(train_json[item])
       #print(train_json[item]['height'])
       new_dict['height'] = [max(list_sum(train_json[item]['top'], train_json[item]['height'])) - min(train_json[item]['top'])]
       new_dict['label'] = [1]
       new_dict['left'] = [max(0,min(train_json[item]['left']))]
       new_dict['top'] = [max(0, min(train_json[item]['top']))]
       new_dict['width'] = [max(list_sum(train_json[item]['width'], train_json[item]['left'])) - min(train_json[item]['left'])]
       #print(new_dict)

       arr = parse_json(new_dict)
      # print(arr.shape[1])
       #print(arr)
       # plt.figure(figsize=(10, 10))
       # plt.subplot(1, arr.shape[1] + 1, 1)
       # plt.imshow(img)
       # plt.show()
       plt.xticks([]);plt.yticks([])

       for idx in range(arr.shape[1]):
          #plt.subplot(1, arr.shape[1] + 1, idx + 2)

          test = img[arr[0, idx]:arr[0, idx] + arr[1, idx], arr[2, idx]:arr[2, idx] + arr[3, idx]]
          # plt.imshow(test)
          # plt.show()
          # plt.savefig()
          item = item.replace('.png', '')
          cropdir = "D:/data/SVHN/val/crop_val/" + item         #存储剪裁图片的路径

          # print(item)
          #print(cropdir + '.png')
          cv2.imwrite(cropdir + '.png',test)
          #plt.title(arr[4, idx])
          # print(arr[4, idx])
          plt.xticks([]);
          plt.yticks([])






