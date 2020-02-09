# CVPR2020_1413

PyTorch implementation for Interpretable Neural Network Decoupling.

## Running Code

In this code, you can run our Resnet-56 model on CIFAR10 dataset. The code has been tested by Python 3.7, [Pytorch 1.3.0](https://pytorch.org/) and CUDA 10.1 on Ubuntu 16.04.

## Running Example

### Train

#### CIFAR-10

##### ResNet-56

```shell
python train.py --net resnet56_architecture_decoupling \
                --pretrained True \
                --checkpoint pth/resnet56.pth \
                --train_dir tmp/resnet56_architecture_decoupling \
                --train_batch_size 128 \
                --learning_rate 0.1 \
                --epochs 200 \
                --schedule 100 150 \
                --Lambda_k 1 \
                --Lambda_m 0.01 \
                --Lambda_s 0.00015
```

##### VGGNet

```shell
python train.py --net vggnet_architecture_decoupling \
                --pretrained True \
                --checkpoint pth/vggnet.pth \
                --train_dir tmp/vggnet_architecture_decoupling \
                --train_batch_size 128 \
                --learning_rate 0.1 \
                --epochs 200 \
                --schedule 100 150 \
                --Lambda_k 1 \
                --Lambda_m 0.04 \
                --Lambda_s 0.0002
```

##### GoogleNet

```shell
python train.py --net googlenet_architecture_decoupling \
                --pretrained True \
                --checkpoint pth/googlenet.pth \
                --train_dir tmp/googlenet_architecture_decoupling \
                --train_batch_size 128 \
                --learning_rate 0.1 \
                --epochs 200 \
                --schedule 100 150 \
                --Lambda_k 1 \
                --Lambda_m 0.006 \
                --Lambda_s 0.00005
```

#### ImageNet

Modifying the 'train_image_dir' and 'val_image_dir' in dataset/imagenet.py to the path of ImageNet dataset. The pretrained model from pytorch model zoo.

##### ResNet-18

```shell
python train.py --net resnet18_architecture_decoupling \
                --pretrained True \
                --checkpoint pth/resnet18.pth \
                --dataset imagenet
                --train_dir tmp/resnet18_architecture_decoupling \
                --train_batch_size 256 \
                --learning_rate 0.01 \
                --epochs 120 \
                --schedule 30 60 90 \
                --Lambda_k 1 \
                --Lambda_m 0.005 \
                --Lambda_s 0.01 \
                --target_com 0.6
```

##### VGG-16

```shell
python train.py --net vgg16_architecture_decoupling \
                --pretrained True \
                --checkpoint pth/vgg16.pth \
                --dataset imagenet
                --train_dir tmp/vgg16_architecture_decoupling \
                --train_batch_size 64 \
                --learning_rate 0.01 \
                --epochs 120 \
                --schedule 30 60 90 \
                --Lambda_k 1 \
                --Lambda_m 0.01 \
                --Lambda_s 0.01 \
                --target_com 0.5
```
