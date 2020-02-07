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
