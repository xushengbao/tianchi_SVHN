#处理AB-Darknet检测输出的json文件
#第一行为坐标开始
import codecs
a = 'D:/data/SVHN/test/yolo3/result.txt'
file = open(a,mode='r', encoding='utf-8')  # 如果你的x.txt文件不在python的路径下,那么必须用绝对路径
line = file.readline()  # 这时候l1的结果是一个list,每个元素是文件的每一行,包括转行符号'/n'



i = 100000
while line:
    #if '.png' in line:

    #print(line)
    if '.png' in line:
        #print(line)
        i+=1
        #j += 1
        a1 = (i // 10000) % 10
        a2 = (i // 1000) % 10
        a3 = (i // 100) % 10
        a4 = (i // 10) % 10
        a5 = i % 10
        a = '0' + str(a1) + str(a2) + str(a3) + str(a4) + str(a5) + '.txt'
        c = 'D:/data/SVHN/test/yolo3/qushu/' + a
        # print(c)
        qushu = open(c,'w')
        qushu.close()

    else:

        a1 = (i // 10000) % 10
        a2 = (i // 1000) % 10
        a3 = (i // 100) % 10
        a4 = (i // 10) % 10
        a5 = i % 10
        a = '0' + str(a1) + str(a2) + str(a3) + str(a4) + str(a5) + '.txt'
        c = 'D:/data/SVHN/test/yolo3/qushu/' + a
        # print(c)
        qushu = open(c, 'a+')

        line = line.replace("%",'')
        line = line.replace('(left_x:','')
        line = line.replace('top_y:', '')
        line = line.replace('width:', '')
        line = line.replace('height', '')
        line = line.replace(':', ' ')
        line = line.replace(')', ' ')

        l1 = line.split()
        l2 = l1[:6]
        #print(l2)
        l3 = [str(l2[0]) + ' ' + str(l2[1]) + ' ' + str(l2[2]) + ' ' + str(l2[3]) + ' ' + str(l2[4]) + ' ' + str(l2[5])]
        #print(l3)
        qushu.write("".join(l3) + '\n')
        qushu.close()

    line = file.readline()
# print(j)
file.close()


