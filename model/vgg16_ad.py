#coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math
import time
from utils.utils import AD_Block

class BasicBlock_vgg16(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlock_vgg16, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        return out

class BasicBlock_vgg16_ad(nn.Module):
    def __init__(self, in_planes, planes):
        super(BasicBlock_vgg16_ad, self).__init__()

        self.ad_block = AD_Block(in_planes, planes, 1000)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1)

    def forward(self, x):
        self.mask = self.ad_block(x)
        self.out = F.relu(self.conv1(x))
        out = self.mask * self.out
        return out

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        # self._initialize_weights()

    def forward(self, x):
        x, ms, os, cs = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x, ms, os, cs

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

class VGG_s(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG_s, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        # self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if i == 0:
                layers += [BasicBlock_vgg16(in_channels, v)]
            else:
                if batch_norm:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [BasicBlock_vgg16_ad(in_channels, v)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16_architecture_decoupling(pretrained=True, checkpoint=None, **kwargs):
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        data = torch.load(checkpoint)
        model.load_state_dict(data)
    return model
