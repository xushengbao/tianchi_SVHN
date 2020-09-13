import os, sys, glob, shutil, json
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import cv2

from PIL import Image
import numpy as np

from tqdm import tqdm, tqdm_notebook

# %pylab inline

import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
#from warpctc_pytorch import CTCLoss
from torchcontrib.optim import SWA

class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl) + (5 - len(lbl)) * [10]
        return img, torch.from_numpy(np.array(lbl[:5]))

    def __len__(self):
        return len(self.img_path)

#train_path = glob.glob('/data/code/bisai/SVHN/data/train/mchar_train/*.png')
train_path = glob.glob('D:/data/SVHN/train/crop_train/*.png')
train_path.sort()
#train_json = json.load(open('/data/code/bisai/SVHN/data/train/mchar_train.json'))
train_json = json.load(open('D:/data/SVHN/train/mchar_train.json'))
train_label = [train_json[x]['label'] for x in train_json]
#print(len(train_path), len(train_label))

train_loader = torch.utils.data.DataLoader(
    SVHNDataset(train_path, train_label,
                transforms.Compose([
                    transforms.Resize((64, 98)),
                    transforms.RandomCrop((60, 90)),
                    transforms.ColorJitter(0.3, 0.3, 0.2),
                    #transforms.ColorJitter(0.6, 0.6, 0.4),
                    #transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #源代码
                    #transforms.Normalize([0.44068253, 0.43229532, 0.43505937], [0.21003874, 0.216583, 0.21854207]) #计算训练集
                    transforms.Normalize([0.43342078, 0.4433036, 0.47769606], [0.19589804, 0.19865277, 0.19502434])  #30000 60*90
    ])),
    batch_size=20,
    shuffle=True,
    num_workers=0,
)

val_path = glob.glob('D:/data/SVHN/val/crop_val/*.png')
val_path.sort()
val_json = json.load(open('D:/data/SVHN/val/mchar_val.json'))
val_label = [val_json[x]['label'] for x in val_json]
#print(len(val_path), len(val_label))

val_loader = torch.utils.data.DataLoader(
    SVHNDataset(val_path, val_label,
                transforms.Compose([
                    transforms.Resize((60, 90)),
                    #transforms.CenterCrop((60, 120)),
                    #transforms.RandomCrop((60, 120)),
                    #transforms.ColorJitter(0.3, 0.3, 0.2),
                    # transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    transforms.Normalize([0.4503194, 0.45206192, 0.4714762], [0.21428213, 0.2206297, 0.22306733]) #剪辑成(60，90)后计算所得
    ])),
    batch_size=20,
    shuffle=False,
    num_workers=0,
)


class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()

        model_conv = models.resnet101(pretrained=True)
        #model_conv = models.resnet50(pretrained=True)
        # model_conv = models.densenet121(pretrained=True)
        # model_conv.fc = nn.Linear(512, 11)
        # self.conv_ = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=3,
        #                 stride=1, padding=1, bias=True)

        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv

        #         self.fc1 = nn.Linear(512, 11)
        #         self.fc2 = nn.Linear(512, 11)
        #         self.fc3 = nn.Linear(512, 11)
        #         self.fc4 = nn.Linear(512, 11)
        #         self.fc5 = nn.Linear(512, 11)
        self.fc1 = nn.Linear(2048, 11)
        self.fc2 = nn.Linear(2048, 11)
        self.fc3 = nn.Linear(2048, 11)
        self.fc4 = nn.Linear(2048, 11)
        self.fc5 = nn.Linear(2048, 11)

    def forward(self, img):
        feat = self.cnn(img)
        # print(feat.size())
        # feat = self.conv_(feat)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        return c1, c2, c3, c4, c5


def train(train_loader, model, criterion, optimizer, epoch):
    # 切换模型为训练模式
    model.train()
    train_loss = []

    for i, (input, target) in enumerate(train_loader):
        if use_cuda:
            input = input.cuda()
            target = target.cuda()
            target = target.long()  #不加会在loss计算过程中报错

        c0, c1, c2, c3, c4 = model(input)
        loss = 0.3 * criterion(c0, target[:, 0]) + \
               0.3 * criterion(c1, target[:, 1]) + \
               0.2 * criterion(c2, target[:, 2]) + \
               0.1 * criterion(c3, target[:, 3]) + \
               0.1 * criterion(c4, target[:, 4])

        # loss /= 6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
    return np.mean(train_loss)


def validate(val_loader, model, criterion):
    # 切换模型为预测模型
    model.eval()
    val_loss = []

    # 不记录模型梯度信息
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if use_cuda:
                input = input.cuda()
                target = target.cuda()
                target = target.long()  # 不加会在loss计算过程中报错

            c0, c1, c2, c3, c4 = model(input)
            loss = 0.3 * criterion(c0, target[:, 0]) + \
                   0.3 * criterion(c1, target[:, 1]) + \
                   0.2 * criterion(c2, target[:, 2]) + \
                   0.1 * criterion(c3, target[:, 3]) + \
                   0.1 * criterion(c4, target[:, 4])
            # loss /= 6
            val_loss.append(loss.item())
    return np.mean(val_loss)


def predict(test_loader, model, tta=10):
    model.eval()
    test_pred_tta = None

    # TTA 次数
    for _ in range(tta):
        test_pred = []

        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                if use_cuda:
                    input = input.cuda()

                c0, c1, c2, c3, c4 = model(input)
                if use_cuda:
                    output = np.concatenate([
                        c0.data.cpu().numpy(),
                        c1.data.cpu().numpy(),
                        c2.data.cpu().numpy(),
                        c3.data.cpu().numpy(),
                        c4.data.cpu().numpy()], axis=1)
                else:
                    output = np.concatenate([
                        c0.data.numpy(),
                        c1.data.numpy(),
                        c2.data.numpy(),
                        c3.data.numpy(),
                        c4.data.numpy()], axis=1)

                test_pred.append(output)

        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta


test_path = glob.glob('D:/data/SVHN/test/crop_test/*.png')
test_path.sort()
#test_json = json.load(open('../input/test_a.json'))
test_label = [[1]] * len(test_path)
#print(len(test_path), len(test_label))

test_loader = torch.utils.data.DataLoader(
    SVHNDataset(test_path, test_label,
                transforms.Compose([
                    transforms.Resize((60, 90)),
                    #transforms.RandomCrop((60, 120)),
                    #transforms.Resize((64, 128)),
                    #transforms.CenterCrop((60, 120)),
                    # transforms.ColorJitter(0.3, 0.3, 0.2),
                    # transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    transforms.Normalize([0.42726642, 0.4270713, 0.4425958], [0.19372733, 0.19580664, 0.1984615])
    ])),
    batch_size=20,
    shuffle=False,
    num_workers=0,
)



if __name__ == '__main__':
    model = SVHN_Model1()
    # model.load_state_dict(torch.load('/data/code/bisai/SVHN/code/four/version_4/model2_acc.pt'))
    criterion = nn.CrossEntropyLoss()
    # criterion = CTCLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    scheduler = SWA(optimizer, swa_start= 10, swa_freq=5, swa_lr=0.0005)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, last_epoch=-1)
    best_loss = 1000.0
    best_acc = 0.63

    # 是否使用GPU
    torch.cuda.current_device()
    use_cuda = True
    if use_cuda:
        model = model.cuda()
    for epoch in range(50):
        #scheduler.step()
        scheduler.zero_grad()
        train_loss = train(train_loader, model, criterion, optimizer, epoch)
        val_loss = validate(val_loader, model, criterion)
        scheduler.step()

        val_label = [''.join(map(str, x)) for x in val_loader.dataset.img_label]
        val_predict_label = predict(val_loader, model, 3)
        val_predict_label = np.vstack([
            val_predict_label[:, :11].argmax(1),
            val_predict_label[:, 11:22].argmax(1),
            val_predict_label[:, 22:33].argmax(1),
            val_predict_label[:, 33:44].argmax(1),
            val_predict_label[:, 44:55].argmax(1),
        ]).T
        val_label_pred = []
        for x in val_predict_label:
            val_label_pred.append(''.join(map(str, x[x != 10])))

        val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label))

        print('Epoch: {0}, Train loss: {1} \t Val loss: {2}'.format(epoch, train_loss, val_loss))
        print('Val Acc', val_char_acc)
        # 记录下验证集精度
        if val_loss < best_loss:
            best_loss = val_loss
            print(best_loss)
            print(val_char_acc)
            # print('Find better model in Epoch {0}, saving model.'.format(epoch))
            torch.save(model.state_dict(), 'D:/data/SVHN/tijiao_CNN/model/model2_loss.pt')

        if val_char_acc > best_acc:
            best_acc = val_char_acc
            print(best_acc)
            print(val_loss)
            # print('Find better model in Epoch {0}, saving model.'.format(epoch))
            torch.save(model.state_dict(), 'D:/data/SVHN/tijiao_CNN/model/model2_acc.pt')
    scheduler.swap_swa_sgd()

    print("best_loss = {}".format(best_loss))
    print("best_acc = {}".format(best_acc))

    #加载保存的最优模型
    # model.load_state_dict(torch.load('D:/data/SVHN/tijiao_CNN/model/model2_loss.pt'))
    #
    # test_predict_label = predict(test_loader, model, 5)
    # print(test_predict_label.shape)
    #
    # test_label = [''.join(map(str, x)) for x in test_loader.dataset.img_label]
    # test_predict_label = np.vstack([
    #     test_predict_label[:, :11].argmax(1),
    #     test_predict_label[:, 11:22].argmax(1),
    #     test_predict_label[:, 22:33].argmax(1),
    #     test_predict_label[:, 33:44].argmax(1),
    #     test_predict_label[:, 44:55].argmax(1),
    # ]).T
    #
    # test_label_pred = []
    # for x in test_predict_label:
    #     test_label_pred.append(''.join(map(str, x[x != 10])))
    #
    # import pandas as pd
    #
    # df_submit = pd.read_csv('D:/data/SVHN/test/mchar_sample_submit_A.csv')
    # df_submit['file_code'] = test_label_pred
    # df_submit.to_csv('D:/data/SVHN/tijiao_CNN/model/submit.csv', index=None)



