# -*- coding:utf-8 -*-
#批量更改图片编号
import os

class ImageRename():
 def __init__(self):
     self.path = 'D:/data/SVHN/train+val+test/mchar_test_a'

 def rename(self):
     filelist = os.listdir(self.path)
     total_num = len(filelist)

     i = 40000
     # file = open('D:/data/SVHN/test/mchar_test_a.txt', 'w')
     for item in filelist:
         if item.endswith('.png'):
             
             src = os.path.join(os.path.abspath(self.path), item)
             # print(item)
             # file.write(item)
             # file.write('\n')
             dst = os.path.join(os.path.abspath(self.path), '0' + format(str(i), '0>3s') + '.png')
             os.rename(src, dst)
             print('converting %s to %s ...' % (src, dst))
         i = i + 1
     print('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
 newname = ImageRename()
 newname.rename()