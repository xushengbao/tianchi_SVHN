# -*- coding:utf-8 -*-
# yolo3检测结果按边框位置排序后，处理重复边框。重复边框取置信度高的边框
import os
import pandas as pd


class ImageRename():
    def __init__(self):
        self.path = 'D:/data/SVHN/test/yolo3/qushu/'
        #self.path = 'D:/data/SVHN/test/input/paixu/'

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)

        #i = 40000
        # file = open('D:/data/SVHN/test/mchar_test_a.txt', 'w')
        #c = 'D:/data/SVHN/test/input/zhengli/tijiao_2.txt'
        #file = open(c, 'w')
        df_submit = pd.read_csv('D:/data/SVHN/test/input0/zhengli/mchar_sample_submit_A.csv')
        l4 = []
        chongfu = 0
        chongfu0 = 0
        chongfu1 = 0
        chongfu2 = 0
        chongfu3 = 0
        chongfu4 = 0
        for item in filelist:
            if item.endswith('.txt'):
                #src = os.path.join(os.path.abspath(self.path), item)
                #print(item)
                a = self.path + '/' + item

                #b = 'D:/data/SVHN/test/input/tijiao' + item
                b = 'D:/data/SVHN/test/yolo3/chongfu.txt'

                #print(a, b)
                # with open(b,'w+') as w:
                #     while True:
                #         sorted_lines=sorted(open(a), key=lambda s: s.split()[2])
                #         w.write("".join(sorted_lines))
                #         break

                with open(b, 'a+') as w:
                    while True:

                        f = open(a)  # 如果你的x.txt文件不在python的路径下,那么必须用绝对路径
                        l1 = f.readlines()  # 这时候l1的结果是一个list,每个元素是文件的每一行,包括转行符号'/n'
                        #print(l1)
                        l1 = [x.split(' ') for x in l1]
                        l1 = [[x[0], x[1], x[2].replace('/n', '')] for x in l1]  # 这里去掉了每行的'/n'符号
                        # print(l1)
                        l2 = [[int(x[0]), float(x[1]), float(x[2])] for x in l1]
                        #print(l2)
                        # b = l2[0]
                        l3 = []
                        # c = b[2] - l3[1][2]
                        # l3.append(l2[0])
                        #res = [0, 0, 0]
                        if len(l2) != 0:
                            res = l2[0]
                            #print(res)
                            l3.append(res)
                        for x in l2[1:]:

                            # print(x)
                            if x[2] - res[2] >= 3.5:
                                l3.append(x)
                                res = x
                            else:
                                distance = x[2] - res[2]
                                chongfu += 1
                                # if distance == 0:
                                #     chongfu0 += 1
                                if 0<=distance <= 1:
                                    chongfu1 += 1
                                if 1<distance <= 2:
                                    chongfu2 += 1
                                if 2<distance <= 3:
                                    chongfu3 += 1
                                if 3<distance <= 4:
                                    chongfu4 += 1
                                w.write(a + '\n')
                                w.write(str(distance) + '\n')
                                #print(a)
                                if x[1] > res[1]:
                                    if len(l3)!=0:
                                        del l3[-1]
                                        l3.append(x)
                                        res = x

                                    # del l3[-1]
                                    # l3.append(x)
                                    # res = x
                            # res = x
                            #print(res)
                        #print(l3)
                        l3 = [str(x[0]) for x in l3]
                        l4.append(''.join(l3))
                        #print(l4)
                        # df_submit[wenjian]['file_code'] = l4
                        # df_submit.to_csv('D:/data/SVHN/test/yolo3/quchong.csv', index=None)
                        f.close()

                        #file.write("".join(l3) + '\n')
                        break
        print(l4)
        print("chongfu:" + str(chongfu))
        print("chongfu0:" + str(chongfu0))
        print("chongfu1:" + str(chongfu1))
        print("chongfu2:" + str(chongfu2))
        print("chongfu3:" + str(chongfu3))
        print("chongfu4:" + str(chongfu4))
        df_submit['file_code'] = l4
        df_submit.to_csv('D:/data/SVHN/test/yolo3/submit.csv', index=None)

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
