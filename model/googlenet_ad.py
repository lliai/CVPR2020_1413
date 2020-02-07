import torch
import torch.nn as nn
import torch.nn.init as init
from utils.utils import AD_Block

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)


class Inception(nn.Module):
    def __init__(self, in_planes, kernel_1_x, kernel_3_in, kernel_3_x, kernel_5_in, kernel_5_x, pool_planes, num_classes=10):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.ad_Block1 = AD_Block(in_planes, kernel_1_x, num_classes)
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_1_x, kernel_size=1),
            nn.BatchNorm2d(kernel_1_x),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.ad_Block2_1 = AD_Block(in_planes, kernel_3_in, num_classes)
        self.ad_Block2_2 = AD_Block(kernel_3_in, kernel_3_x, num_classes)
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_3_in, kernel_size=1),
            nn.BatchNorm2d(kernel_3_in),
            nn.ReLU(True),
            nn.Conv2d(kernel_3_in, kernel_3_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_3_x),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.ad_Block3_1 = AD_Block(in_planes, kernel_5_in, num_classes)
        self.ad_Block3_2 = AD_Block(kernel_5_in, kernel_5_x, num_classes)
        self.ad_Block3_3 = AD_Block(kernel_5_x, kernel_5_x, num_classes)
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_5_in, kernel_size=1),
            nn.BatchNorm2d(kernel_5_in),
            nn.ReLU(True),
            nn.Conv2d(kernel_5_in, kernel_5_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(True),
            nn.Conv2d(kernel_5_x, kernel_5_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernel_5_x),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.ad_Block4 = AD_Block(in_planes, pool_planes, num_classes)
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        self.mask1 = self.ad_Block1(x)
        self.out1 = self.b1(x)
        y1 = self.out1 * self.mask1

        self.mask2_1 = self.ad_Block2_1(x)
        self.out2_1 = self.b2[0:3](x)
        out = self.mask2_1 * self.out2_1
        self.mask2_2 = self.ad_Block2_2(out)
        self.out2_2 = self.b2[3:](out)
        y2 = self.mask2_2 * self.out2_2

        self.mask3_1 = self.ad_Block3_1(x)
        self.out3_1 = self.b3[0:3](x)
        out = self.mask3_1 * self.out3_1
        self.mask3_2 = self.ad_Block3_2(out)
        self.out3_2 = self.b3[3:6](out)
        out = self.mask3_2 * self.out3_2
        self.mask3_3 = self.ad_Block3_3(out)
        self.out3_3 = self.b3[6:](out)
        y3 = self.mask3_3 * self.out3_3

        self.mask4 = self.ad_Block4(x)
        self.out4 = self.b4(x)
        y4 = self.out4 * self.mask4
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32, num_classes=num_classes)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64, num_classes=num_classes)

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64, num_classes=num_classes)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64, num_classes=num_classes)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64, num_classes=num_classes)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64, num_classes=num_classes)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128, num_classes=num_classes)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128, num_classes=num_classes)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128, num_classes=num_classes)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, num_classes)

        self.apply(_weights_init)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.max_pool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.max_pool(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def googlenet_architecture_decoupling(pretrained=False, checkpoint=None):
    model = GoogLeNet()
    if pretrained:
        model.load_state_dict(torch.load(checkpoint), strict=False)
    return model

def googlenet_architecture_decoupling_cifar100(pretrained=False, checkpoint=None):
    model = GoogLeNet(100)
    if pretrained:
        model.load_state_dict(torch.load(checkpoint), strict=False)
    return model