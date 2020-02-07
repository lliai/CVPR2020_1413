# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class semhash(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v1, v2, training=True):
        index = torch.randint(low=0, high=v1.shape[1], size=[int(v1.shape[1]/2)]).long()
        v1[index] = v2[index]
        return v1

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class AD_Block(nn.Module):

    def __init__(self, in_channel, out_channel, num_classes=10, T=4):
        super(AD_Block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.fc1 = nn.Linear(in_channel, int(in_channel/T))
        self.fc2 = nn.Linear(int(in_channel/T), out_channel)

        self.fc_class = nn.Linear(out_channel, num_classes, bias=False)

        self.training = True

        # init.kaiming_normal(self.fc1.weight)
        self.fc1.weight.data.fill_(0)
        init.kaiming_normal(self.fc2.weight)
        self.fc1.bias.data.fill_(1)
        self.fc2.bias.data.fill_(1)

        nn.init.kaiming_normal_(self.fc_class.weight)

    def forward(self, input):
        x = F.avg_pool2d(input, input.shape[3])
        x = x.view(x.size(0), -1)
        y = F.relu(self.fc1(x))
        self.y = self.fc2(y)

        if self.training:
            g = torch.randn(self.y.shape).cuda()
            g = g + self.y
            v1 = torch.max(torch.min(1.2 * torch.sigmoid(g) - 0.1, torch.Tensor([1]).cuda()), torch.Tensor([0]).cuda())
            v2 = (g > 0).float()
            predict_bin = semhash.apply(v1, v2, self.training)
        else:
            predict_bin = (self.y > 0).float()

        return predict_bin.unsqueeze(2).unsqueeze(3)

    def cforward(self, input):
        y = self.fc_class(input)
        return y

    def train(self, mode=True):
        self.training = mode