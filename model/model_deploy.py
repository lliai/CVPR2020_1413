# coding: utf-8

from .resnet_cifar import resnet56, resnet56_cifar100
from .resnet_cifar_ad import resnet20_architecture_decoupling,resnet56_architecture_decoupling
from .resnet_imagenet import resnet18
from .resnet_imagenet_ad import resnet18_architecture_decoupling
from .googlenet import googlenet, googlenet_cifar100
from .googlenet_ad import googlenet_architecture_decoupling, googlenet_architecture_decoupling_cifar100
from .vgg_cifar import vggnet, vggnet_cifar100
from .vgg_cifar_ad import vggnet_architecture_decoupling, vggnet_architecture_decoupling_cifar100
from .vgg16_ad import vgg16_architecture_decoupling
from .vgg16 import vgg16