1.最好成绩采用的模型：YOLOV3检测算法，采用AB版DarkNet（py文件夹不包括该部分）；
2.py文件夹里面的代码主要为数据处理（如，转化检测需要的格式，以及最后的检测结果去重等）；
3.单个的YOLOV3的检测结果并不高，为此需要提高检测框的准度，以及去除重复检测框，这样即使是YOLOV3的单模型也能有不错的精度；
4.YOLOV3的代码参见：https://github.com/AlexeyAB/darknet
5.weights文件夹，里面为训练后得到的权重文件；
6.submit.csv为补全YOLOV3检测结果的后的提交文件（由于会出现部分图片无法检测的情况，于是用其它模型的结果进行补全，如用定长的结果）。


联系方式 -电话：17729831853；邮箱：chuanliu@stu.scu.edu.cn
审核如有任何疑问，麻烦您通过以上方式联系我，谢谢！

py文件简介：
labelmeTOcoco.py    #把labelme数据转为coco格式的数据
cocoTOvoc.py           #把coco格式的数据集转化为voc格式
jisuan.py                    #多次计算得到kmeans聚类后的均值anchor
buquan.py                 #补全yolo3未检测出来的结果，用定长或者其他比较好的模型，进行结果的赋值
center.py                   #处理AB-Darknet检测输出的json文件
compute_mean.py     #计算自己图像数据集归一化的均值等
juli.py                         # 统计训练数据集边框的left间隔距离
tijiao.py                      # yolo3检测结果按边框位置排序后，处理重复边框。重复边框取置信度高的边框
tijiao_CNN.py             #定长模型
tijiao_csv.py                # 对yolo3检测结果的txt文件转为提交格式的csv
tijiao_noQuChong.py  # yolo3检测结果按边框位置排序后，不处理重复边框。
xgb.py                         #处理街景数据中，训练与验证集合并后的命名
zhunquelv.py               #比较街景识别比赛中不同提交结果的相似度，大致预测精度
