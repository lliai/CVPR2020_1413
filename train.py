# coding: utf-8

import torch
import torch.nn as nn
import argparse
import importlib
from tensorboardX import SummaryWriter
import pdb

from utils import base

parser = argparse.ArgumentParser(description='Weight Decay Experiments')
parser.add_argument('--dataset', dest='dataset', help='training dataset', default='cifar10', type=str)
parser.add_argument('--net', dest='net', help='training network', default='resnet20', type=str)
parser.add_argument('--pretrained', dest='pretrained', help='whether use pretrained model', default=False, type=bool)
parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint dir', default=None, type=str)
parser.add_argument('--train_dir', dest='train_dir', help='training data dir', default="tmp", type=str)
parser.add_argument('--save_best', dest='save_best', help='whether only save best model', default=True, type=bool)

parser.add_argument('--train_batch_size', dest='train_batch_size', help='training batch size', default=64, type=int)
parser.add_argument('--test_batch_size', dest='test_batch_size', help='test batch size', default=50, type=int)

parser.add_argument('--learning_rate', dest='learning_rate', help='learning rate', default=0.1, type=float)
parser.add_argument('--momentum', dest='momentum', help='momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', dest='weight_decay', help='weight decay', default=1e-4, type=float)
parser.add_argument('--epochs', dest='epochs', help='epochs', default=200, type=int)
parser.add_argument('--schedule', dest='schedule', help='Decrease learning rate',default=[100, 150],type=int,nargs='+')
parser.add_argument('--gamma', dest='gamma', help='gamma', default=0.1, type=float)

parser.add_argument('--Lambda_k', dest='Lambda_k', help='kl Lambda', default=0, type=float)
parser.add_argument('--Lambda_s', dest='Lambda_s', help='sparse Lambda', default=0, type=float)
parser.add_argument('--Lambda_m', dest='Lambda_m', help='mutual information Lambda', default=0, type=float)
parser.add_argument('--target_com', dest='target_com', help='target compression ratio', default=0, type=float)

args = parser.parse_args()

if __name__ == "__main__":
    print(args)
    model = importlib.import_module("model.model_deploy").__dict__[args.net](args.pretrained, args.checkpoint)

    train_loader, test_loader = importlib.import_module("dataset."+args.dataset).__dict__["load_data"](
        args.train_batch_size, args.test_batch_size)

    writer = SummaryWriter(args.train_dir)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(filter(lambda i: i.requires_grad, model.parameters()), args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    best_acc = 0
    for i in range(args.epochs):
        base.train_architecture_decouple(train_loader, model, criterion, optimizer, i, writer, args.Lambda_k, args.Lambda_s,
                                         args.Lambda_m, args.target_com, args.net)

        top1_acc, top5_acc = base.validate_architecture_decouple(test_loader, model, criterion, args.net)

        lr_scheduler.step()

        if best_acc < top1_acc:
            if args.save_best:
                torch.save(model.state_dict(), args.train_dir+"/model_best.pth")
            best_acc = top1_acc
        if not args.save_best:
            torch.save(model.state_dict(), args.train_dir + "/model_"+str(i)+".pth")

        writer.add_scalar('val-acc', top1_acc, i)
        writer.add_scalar('val-top5-acc', top5_acc, i)

    print("best acc: {:.2f}".format(best_acc))
