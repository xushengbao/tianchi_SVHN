# -*- coding:utf-8 -*-
# 对yolo3检测结果按边框位置排序
import numpy as np
import os
img_path='D:/data/SVHN/test/yolo3/paixu'

# img_list=os.listdir(img_path)
# img_list.sort()
# img_list.sort(key = lambda x: int(x[:-6])) ##文件名按数字排序




class ImageRename():
    def __init__(self):
        self.path = 'D:/data/SVHN/test/yolo3/paixu'

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)

        #i = 40000
        # file = open('D:/data/SVHN/test/mchar_test_a.txt', 'w')
        # c = 'D:/data/SVHN/test/input/zhengli/tijiao_1.txt'
        # file = open(c, 'w')
        for item in filelist:
            if item.endswith('.txt'):
                #src = os.path.join(os.path.abspath(self.path), item)
                # print(item)
                a = self.path + '/' + item
                b = 'D:/data/SVHN/test/yolo3/qushu/' + item
                #b = 'D:/data/SVHN/test/input/tijiao_1.txt'

                #print(a, b)
                # with open(b,'w+') as w:
                #     while True:
                #         sorted_lines=sorted(open(a), key=lambda s: s.split()[2])
                #         w.write("".join(sorted_lines))
                #         break

                with open(b, 'w+') as w:
                    while True:

                        f = open(a)  # 如果你的x.txt文件不在python的路径下,那么必须用绝对路径
                        l1 = f.readlines()  # 这时候l1的结果是一个list,每个元素是文件的每一行,包括转行符号'/n'
                        l1 = [x.split(' ') for x in l1]
                        #l1 = [[x[0], x[1], x[2], x[3], x[4], x[5].replace('/n', '')] for x in l1]  # 这里去掉了每行的'/n'符号
                        #print(l1)
                        l1 = [[x[0], x[1], x[2].replace('/n', '')] for x in l1]
                        l2 = [[int(x[0]), float(x[1]), float(x[2])] for x in l1]
                        l2.sort(key=lambda x: x[2])
                        # print(l2)
                        l2 = [str(x[0]) + ' ' + str(x[1]) + ' ' + str(x[2]) + '\n' for x in l2]
                        #l2 = [str(x[0]) for x in l2]
                        # print(l2)
                        f.close()

                        w.write("".join(l2))
                        #file.write("".join(l2)  + '\n' )
                        break

if __name__ == '__main__':
    newname = ImageRename()
    newname.rename()




# with open('D:/data/SVHN/test/text.txt','w+') as w:
#     while True:
#         # sorted_lines=sorted(open('D:/data/SVHN/test/000259.png.txt'), key=lambda s: s.split(' ')[2])
#         # print(sorted_lines)
#         # #print(sorted_lines[2])
#
#
#         f = open('D:/data/SVHN/test/000259.png.txt')  # 如果你的x.txt文件不在python的路径下,那么必须用绝对路径
#         l1 = f.readlines()# 这时候l1的结果是一个list,每个元素是文件的每一行,包括转行符号'/n'
#         l1 = [x.split(' ') for x in l1]
#         l1 = [[x[0], x[1],x[2].replace('/n', '')] for x in l1] # 这里去掉了每行的'/n'符号
#         #print(l1)
#         l2 = [[int(x[0]), float(x[1]),float(x[2])] for x in l1]
#         print(l2)
#         # b = l2[0]
#         l3 = []
#         # c = b[2] - l3[1][2]
#         # l3.append(l2[0])
#         res = [0, 0, 0]
#         for x in l2:
#             #print(x)
#             if x[2] - res[2] >4:
#                 l3.append(x)
#                 res = x
#             else:
#                 if x[1] > res[1]:
#                     del l3[-1]
#                     l3.append(x)
#                     res = x
#             #res = x
#             print(res)
#         print(l3)
#
#
#
#             # print(a1)
#         #l2.sort(key =lambda x:x[2])
#         l3 = [[x[0], x[1], x[2]] for x in l3]
#         #print(l2)
#         #l3 = [str(x[0]) + ' ' + str(x[1]) + ' ' + str(x[2]) + '\n' for x in l3]
#         l3 = [str(x[0])for x in l3]
#         print(l3)
#         f.close()
#         #
#         w.write("".join(l3))
#         break
