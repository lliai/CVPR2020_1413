# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import pdb
from functools import reduce
import operator
import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda(async=True)
        target = target.cuda(async=True)
        input.requires_grad_()
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            writer.add_scalar('loss', losses.val, i + epoch * len(train_loader))
            writer.add_scalar('acc', top1.val, i + epoch * len(train_loader))
            writer.add_scalar('top5-acc', top5.val, i + epoch * len(train_loader))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

def train_architecture_decouple(train_loader, model, criterion, optimizer, epoch, writer, Lambda_k, Lambda_s, Lambda_m, target_com, net):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kl_losses = AverageMeter()
    s_losses = AverageMeter()
    m_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda(async=True)
        target = target.cuda(async=True)
        input.requires_grad_()
        # compute output
        output = model(input)
        loss = criterion(output, target)

        kl_loss = 0
        s_loss = 0
        m_loss = 0
        if "resnet56" in net:
            for k in range(1, 4):
                for j in range(9):
                    out1 = model.__getattr__("layer" + str(k))[j].out1
                    out1 = F.avg_pool2d(out1, out1.shape[3]).view(input.shape[0], -1)
                    mask1 = model.__getattr__("layer" + str(k))[j].mask1.squeeze(3).squeeze(2)

                    cls1 = model.__getattr__("layer" + str(k))[j].ad_block1.cforward(mask1)

                    m_loss += criterion(cls1, target)
                    kl_loss += F.kl_div(F.log_softmax(mask1, 1), F.softmax(out1, 1))
                    s_loss += torch.abs(torch.mean(torch.norm(mask1, 1, 1)) - target_com*mask1.shape[1])

                    out2 = model.__getattr__("layer" + str(k))[j].out2
                    out2 = F.avg_pool2d(out2, out2.shape[3]).view(input.shape[0], -1)
                    mask2 = model.__getattr__("layer" + str(k))[j].mask2.squeeze(3).squeeze(2)

                    cls2 = model.__getattr__("layer" + str(k))[j].ad_block2.cforward(mask2)

                    m_loss += criterion(cls2, target)
                    kl_loss += F.kl_div(F.log_softmax(mask2, 1), F.softmax(out2, 1))
                    s_loss += torch.abs(torch.mean(torch.norm(mask2, 1, 1)) - target_com*mask2.shape[1])

        elif "resnet20" in net:
            for k in range(1, 4):
                for j in range(3):
                    out = model.__getattr__("layer" + str(k))[j].out
                    out = F.avg_pool2d(out, out.shape[3]).view(input.shape[0], -1)
                    mask = model.__getattr__("layer" + str(k))[j].mask.squeeze(3).squeeze(2)

                    cls = model.__getattr__("layer"+str(k))[j].ad_block.cforward(mask)

                    m_loss += criterion(cls, target)
                    kl_loss += F.kl_div(F.log_softmax(out, 1), F.softmax(mask, 1))
                    s_loss += torch.abs(torch.mean(torch.norm(mask, 1, 1)) - target_com*mask.shape[1])

        elif "vggnet" in net:
            for k in range(20):
                if isinstance(model.features[k], nn.MaxPool2d) or k == 0:
                    continue

                out = model.features[k].out
                out = F.avg_pool2d(out, out.shape[3]).view(input.shape[0], -1)
                mask = model.features[k].mask.squeeze(3).squeeze(2)

                cls = model.features[k].ad_block.cforward(mask)

                m_loss += criterion(cls, target)
                kl_loss += F.kl_div(F.log_softmax(out, 1), F.softmax(mask, 1))
                s_loss += torch.abs(torch.mean(torch.norm(mask, 1, 1)) - target_com*mask.shape[1])

        elif "resnet18" in net:
            for k in range(1, 5):
                for j in range(2):
                    out1 = model.__getattr__("layer" + str(k))[j].out1
                    out1 = F.avg_pool2d(out1, out1.shape[3]).view(input.shape[0], -1)
                    mask1 = model.__getattr__("layer" + str(k))[j].mask1.squeeze(3).squeeze(2)

                    cls1 = model.features[k].ad_block1.cforward(mask1)

                    m_loss += criterion(cls1, target)
                    kl_loss += F.kl_div(F.log_softmax(mask1, 1), F.softmax(out1, 1))
                    s_loss += torch.abs(torch.mean(torch.norm(mask1, 1, 1)) - target_com*mask1.shape[1])

                    out2 = model.__getattr__("layer" + str(k))[j].out2
                    out2 = F.avg_pool2d(out2, out2.shape[3]).view(input.shape[0], -1)
                    mask2 = model.__getattr__("layer" + str(k))[j].mask2.squeeze(3).squeeze(2)
                    cls2 = model.features[k].ad_block2.cforward(mask2)

                    m_loss += criterion(cls2, target)
                    kl_loss += F.kl_div(F.log_softmax(mask2, 1), F.softmax(out2, 1))
                    s_loss += torch.abs(torch.mean(torch.norm(mask2, 1, 1)) - target_com*mask2.shape[1])

                    if model.__getattr__("layer" + str(k))[j].downsample is not None:
                        out3 = model.__getattr__("layer" + str(k))[j].out3
                        out3 = F.avg_pool2d(out3, out3.shape[3]).view(input.shape[0], -1)
                        mask3 = model.__getattr__("layer" + str(k))[j].mask3.squeeze(3).squeeze(2)
                        cls3 = model.features[k].ad_block3.cforward(mask3)

                        m_loss += criterion(cls3, target)
                        kl_loss += F.kl_div(F.log_softmax(mask3, 1), F.softmax(out3, 1))
                        s_loss += torch.abs(torch.mean(torch.norm(mask3, 1, 1)) - target_com*mask3.shape[1])

        elif "vgg16" in net:

            for k in range(17):
                if isinstance(model.features[k], nn.MaxPool2d) or k == 0:
                    continue
                out = model.features[k].out
                out = F.avg_pool2d(out, out.shape[3]).view(input.shape[0], -1)
                mask = model.features[k].mask.squeeze(3).squeeze(2)

                cls = model.features[k].ad_block.cforward(mask)

                m_loss += criterion(cls, target)
                kl_loss += F.kl_div(F.log_softmax(out, 1), F.softmax(mask, 1))
                s_loss += torch.abs(torch.mean(torch.norm(mask, 1, 1)) - target_com*mask.shape[1])

        elif "googlenet" in net:
            layer = ['a3', 'b3', 'a4', 'b4', 'c4', 'd4', 'e4', 'a5', 'b5']
            module = ['1', '2_1', '2_2', '3_1', '3_2', '3_3', '4']
            for k in layer:
                for j in module:
                    out = model.__getattr__(k).__getattribute__("out"+j)
                    out = F.avg_pool2d(out, out.shape[3]).view(input.shape[0], -1)
                    mask = model.__getattr__(k).__getattribute__("mask"+j).squeeze(3).squeeze(2)
                    cls = model.__getattr__(k).__getattr__("ad_block"+j).cforward(mask)
                    m_loss += criterion(cls, target)
                    kl_loss += F.kl_div(F.log_softmax(mask, 1), F.softmax(out, 1))
                    s_loss += torch.abs(torch.mean(torch.norm(mask, 1, 1)) - target_com*mask.shape[1])

        m_loss = Lambda_m * m_loss
        kl_loss = Lambda_k * kl_loss
        s_loss = Lambda_s * s_loss

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        kl_losses.update(kl_loss.data.item(), input.size(0))
        s_losses.update(s_loss.data.item(), input.size(0))
        m_losses.update(m_loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        (loss + kl_loss + s_loss + m_loss).backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            sparse = []
            var = []
            if "resnet56" in net:
                for k in range(1, 4):
                    for j in range(9):
                        mask1 = model.__getattr__("layer" + str(k))[j].mask1
                        sparse.append(round((torch.sum(mask1) / reduce(operator.mul, mask1.size(), 1)).data.item(), 2))

                        mask2 = model.__getattr__("layer" + str(k))[j].mask2
                        sparse.append(round((torch.sum(mask2) / reduce(operator.mul, mask2.size(), 1)).data.item(), 2))

            elif "resnet20" in net:
                for k in range(1, 4):
                    for j in range(3):
                        mask1 = model.__getattr__("layer" + str(k))[j].mask
                        sparse.append(round((torch.sum(mask1) / reduce(operator.mul, mask1.size(), 1)).data.item(), 2))

            elif "vggnet" in net:
                for k in range(20):
                    if isinstance(model.features[k], nn.MaxPool2d) or k == 0:
                        continue
                    mask = model.features[k].mask.squeeze(3).squeeze(2)
                    sparse.append(round((torch.sum(mask) / reduce(operator.mul, mask.size(), 1)).data.item(), 2))

            elif "resnet18" in net:
                for k in range(1, 5):
                    for j in range(2):
                        mask1 = model.__getattr__("layer" + str(k))[j].mask1
                        sparse.append(round((torch.sum(mask1) / reduce(operator.mul, mask1.size(), 1)).data.item(), 2))
                        var.append(round(
                            torch.mean(torch.norm(mask1 - torch.mean(mask1, 0).unsqueeze(0), 1, 1)).data.item(), 2))
                        
                        mask2 = model.__getattr__("layer" + str(k))[j].mask2
                        sparse.append(round((torch.sum(mask2) / reduce(operator.mul, mask2.size(), 1)).data.item(), 2))
                        var.append(round(
                            torch.mean(torch.norm(mask2 - torch.mean(mask2, 0).unsqueeze(0), 1, 1)).data.item(), 2))
                        
                        if model.__getattr__("layer" + str(k))[j].downsample is not None:
                            mask3 = model.__getattr__("layer" + str(k))[j].mask3.squeeze(3).squeeze(2)
                            sparse.append(
                                round((torch.sum(mask3) / reduce(operator.mul, mask3.size(), 1)).data.item(), 2))
                            var.append(round(
                                torch.mean(torch.norm(mask3 - torch.mean(mask3, 0).unsqueeze(0), 1, 1)).data.item(), 2))

            elif "vgg16" in net:
                for k in range(17):
                    if isinstance(model.features[k], nn.MaxPool2d) or k == 0:
                        continue
                    mask = model.features[k].mask.squeeze(3).squeeze(2)
                    sparse.append(
                        round((torch.sum(mask) / reduce(operator.mul, mask.size(), 1)).data.item(), 2))

            elif "googlenet" in net:
                layer = ['a3', 'b3', 'a4', 'b4', 'c4', 'd4', 'e4', 'a5', 'b5']
                module = ['1', '2_1', '2_2', '3_1', '3_2', '3_3', '4']
                for k in layer:
                    for j in module:
                        mask = model.__getattr__(k).__getattribute__("mask" + j).squeeze(3).squeeze(2)
                        sparse.append(round((torch.sum(mask) / reduce(operator.mul, mask.size(), 1)).data.item(), 2))
            print(sparse)

            writer.add_scalar('loss', losses.val, i + epoch * len(train_loader))
            writer.add_scalar('m_loss', m_losses.val, i + epoch * len(train_loader))
            writer.add_scalar('kl_loss', kl_losses.val, i + epoch * len(train_loader))
            writer.add_scalar('s_loss', s_losses.val, i + epoch * len(train_loader))
            writer.add_scalar('acc', top1.val, i + epoch * len(train_loader))
            writer.add_scalar('top5-acc', top5.val, i + epoch * len(train_loader))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'kl_loss {kl_loss.val:.4f} ({kl_loss.avg:.4f})\t'
                  's_Loss {sloss.val:.4f} ({sloss.avg:.4f})\t'
                  'm_loss {closs.val:.4f} ({closs.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, kl_loss=kl_losses, sloss=s_losses, closs=m_losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda(async=True)

            input.requires_grad_()

            # compute output
            torch.cuda.synchronize()
            end = time.time()
            output = model(input)
            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)

            loss = criterion(output, target)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            if i % 100 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {batch_time.avg:.3f}'
              .format(top1=top1, top5=top5, batch_time=batch_time))

    return top1.avg, top5.avg


def validate_architecture_decouple(val_loader, model, criterion, net):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda(async=True)

            input.requires_grad_()

            # compute output
            torch.cuda.synchronize()
            end = time.time()
            output = model(input)
            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)

            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            if i % 100 == 0:
                sparse = []
                var = []

                if "resnet56" in net:
                    for k in range(1, 4):
                        for j in range(9):
                            mask1 = model.__getattr__("layer" + str(k))[j].mask1
                            sparse.append(
                                round((torch.sum(mask1) / reduce(operator.mul, mask1.size(), 1)).data.item(), 2))

                            mask2 = model.__getattr__("layer" + str(k))[j].mask2
                            sparse.append(
                                round((torch.sum(mask2) / reduce(operator.mul, mask2.size(), 1)).data.item(), 2))

                elif "resnet20" in net:
                    for k in range(1, 4):
                        for j in range(3):
                            mask = model.__getattr__("layer" + str(k))[j].mask
                            sparse.append(
                                round((torch.sum(mask) / reduce(operator.mul, mask.size(), 1)).data.item(), 2))

                elif "vggnet" in net:
                    for k in range(20):
                        if isinstance(model.features[k], nn.MaxPool2d) or k == 0:
                            continue
                        mask = model.features[k].mask.squeeze(3).squeeze(2)
                        sparse.append(round((torch.sum(mask) / reduce(operator.mul, mask.size(), 1)).data.item(), 2))

                elif "resnet18" in net:
                    for k in range(1, 5):
                        for j in range(2):
                            mask1 = model.__getattr__("layer" + str(k))[j].mask1
                            sparse.append(
                                round((torch.sum(mask1) / reduce(operator.mul, mask1.size(), 1)).data.item(), 2))

                            mask2 = model.__getattr__("layer" + str(k))[j].mask2
                            sparse.append(
                                round((torch.sum(mask2) / reduce(operator.mul, mask2.size(), 1)).data.item(), 2))

                            if model.__getattr__("layer" + str(k))[j].downsample is not None:
                                mask3 = model.__getattr__("layer" + str(k))[j].mask3.squeeze(3).squeeze(2)
                                sparse.append(
                                    round((torch.sum(mask3) / reduce(operator.mul, mask3.size(), 1)).data.item(), 2))

                elif "googlenet" in net:
                    layer = ['a3', 'b3', 'a4', 'b4', 'c4', 'd4', 'e4', 'a5', 'b5']
                    module = ['1', '2_1', '2_2', '3_1', '3_2', '3_3', '4']
                    for k in layer:
                        for j in module:
                            mask = model.__getattr__(k).__getattribute__("mask" + j).squeeze(3).squeeze(2)
                            sparse.append(
                                round((torch.sum(mask) / reduce(operator.mul, mask.size(), 1)).data.item(), 2))
                            var.append(
                                round(torch.mean(torch.norm(mask - torch.mean(mask, 0).unsqueeze(0), 1, 1)).data.item(),
                                      2))

                print(sparse)
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {batch_time.avg:.3f}'
              .format(top1=top1, top5=top5, batch_time=batch_time))

    return top1.avg, top5.avg
