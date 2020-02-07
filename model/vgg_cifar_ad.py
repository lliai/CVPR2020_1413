# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
import time
import pdb
from utils.utils import AD_Block

class BasicBlock_vgg(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlock_vgg, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out


class BasicBlock_vgg_ad(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlock_vgg_ad, self).__init__()

        self.ad_block = AD_Block(in_planes, planes, 10)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        self.mask = self.ad_block(x)
        self.out = F.relu(self.bn1(self.conv1(x)))
        out = self.mask * self.out
        return out


class VGG(nn.Module):

    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()

        self.features = features
        self.fc = nn.Linear(512, num_classes)
        # self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, x.shape[3])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


def make_layers(cfg):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if i == 0:
                layers += [BasicBlock_vgg(in_channels, v)]
            else:
                layers += [BasicBlock_vgg_ad(in_channels, v)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


def vggnet_architecture_decoupling(pretrained=True, checkpoint=None):
    model = VGG(make_layers(cfg['E']))
    if pretrained:
        model.load_state_dict(torch.load(checkpoint), strict=False)
    return model

def vggnet_architecture_decoupling_cifar100(pretrained=True, checkpoint=None):
    model = VGG(make_layers(cfg['E']), num_classes=100)
    if pretrained:
        model.load_state_dict(torch.load(checkpoint), strict=False)
    return model