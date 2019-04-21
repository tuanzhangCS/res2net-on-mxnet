#!/bin/bash
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice
# you'd better change setting with your own --data-dir, --depth, --batch-size, --gpus.
# train cifar10
python -u github_train_resnet.py --data-dir data/cifar10 --data-type cifar10 --depth 110 \
       --batch-size 128 --num-classes 10 --num-examples 50000 --gpus=0 --model_load_epoch 4
#python -u train_resnet.py --data-dir data/cifar10 --data-type cifar10 --depth 164 \
#      --batch-size 128 --num-classes 10 --num-examples 50000 --gpus=0,1,2,3,4,5,6,7

## train resnet-50
#python -u train_resnet.py --data-dir data/imagenet --data-type imagenet --depth 50 \
#--batch-size 256 --gpus=0,1,2,3
