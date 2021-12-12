# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Time       : 2021/4/1 20:58
# @Site        : xxx#2L4
# @File         : my_model
# @Software : PyCharm
# @Author   : Dave Liu
# @Email    :
# @Version  : V1.1.0
------------------------------------------------- 
"""
import torch
from torch import nn
from torch import sigmoid
from .basic_module import BasicModule
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import os


class MAWDH(BasicModule):
    def __init__(self, bit, ydim):
        super(MAWDH, self).__init__()

        self.bit = bit
        self.ydim = ydim
        # self.lambd = lambd
        # self.device = device

        self.mean = (torch.tensor([0.485, 0.456, 0.406])).reshape([1, 3, 1, 1]).cuda()
        self.std = (torch.tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])).cuda()

        self.weight = nn.Parameter(torch.from_numpy(np.array([1.0, 1.0, 1.0]).astype('float32')))

        basic_model = models.vgg19(pretrained=True)

        self.feature = basic_model.features
        self.classifier = basic_model.classifier[0: 5]
        # self.avgpool = basic_model.avgpool


        self.hash_layer = nn.Sequential(
            nn.Linear(4096, bit, bias=True),
            nn.Tanh()
        )

        self.txt_module = nn.Sequential(
            nn.Conv2d(1, 8192, kernel_size=(ydim, 1), stride=(1, 1)),
            nn.ReLU(True),
            nn.Dropout2d(0.5),
            nn.Conv2d(8192, 4096, kernel_size=1, stride=(1, 1)),
            nn.ReLU(True),
            nn.Dropout2d(0.5),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(1, 8192, kernel_size=(ydim, 1), stride=(1, 1)),
        )

        self.channel_1 = nn.Sequential(
            nn.Conv2d(8192, 4096, kernel_size=1, stride=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(4096, bit, kernel_size=1, stride=(1, 1)),
            nn.Tanh()
        )

        self.channel_2 = nn.Sequential(
            nn.Conv2d(8192, 2048, kernel_size=1, stride=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(2048, bit, kernel_size=1, stride=(1, 1)),
            nn.Tanh()
        )

        self.channel_3 = nn.Sequential(
            nn.Conv2d(8192, 1024, kernel_size=1, stride=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(1024, bit, kernel_size=1, stride=(1, 1)),
            nn.Tanh()
        )


    def forward(self, x):
        f_x_1 = self.feature((x / 255. - self.mean) / self.std)
        # f_x_2 = self.avgpool(f_x_1)
        f_x_3 = torch.flatten(f_x_1, 1)
        f_x_4 = self.classifier(f_x_3)
        h_x = self.hash_layer(f_x_4)

        return h_x



class Text_net(BasicModule):
    def __init__(self, bit, ydim):
        super(Text_net, self).__init__()
        self.bit = bit
        self.ydim = ydim

        self.weight = nn.Parameter(torch.from_numpy(np.array([0.3, 0.3, 0.4]).astype('float32')))

        self.txt_module = nn.Sequential(
            nn.Conv2d(1, 8192, kernel_size=(ydim, 1), stride=(1, 1)),
            nn.ReLU(True),
            nn.Dropout2d(0.5),
            nn.Conv2d(8192, 4096, kernel_size=1, stride=(1, 1)),
            nn.ReLU(True),
            nn.Dropout2d(0.5),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(1, 8192, kernel_size=(ydim, 1), stride=(1, 1)),
        )

        self.channel_1 = nn.Sequential(
            nn.Conv2d(8192, 4096, kernel_size=1, stride=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(4096, bit, kernel_size=1, stride=(1, 1)),
            nn.Tanh()
        )

        self.channel_2 = nn.Sequential(
            nn.Conv2d(8192, 2048, kernel_size=1, stride=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(2048, bit, kernel_size=1, stride=(1, 1)),
            nn.Tanh()
        )

        self.channel_3 = nn.Sequential(
            nn.Conv2d(8192, 1024, kernel_size=1, stride=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(1024, bit, kernel_size=1, stride=(1, 1)),
            nn.Tanh()
        )

    def forward(self, y):
        feature_1 = self.conv(y.unsqueeze(1).unsqueeze(-1))

        h_y_1 = self.channel_1(feature_1)
        h_y_2 = self.channel_2(feature_1)
        h_y_3 = self.channel_3(feature_1)

        out_hash = self.weight[0] * h_y_1 + self.weight[1] * h_y_2 + self.weight[2] * h_y_3

        return out_hash.squeeze()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    img = torch.randn((64, 3, 224, 224)).cuda()
    model = MAWDH(64, 1386).cuda()
    out = model(img)
    print('ok')

    # text = torch.randn((64, 1386)).cuda()
    # t_model = txt_model(32, 1386).cuda()
    # out = t_model(text)
    # print('ok')