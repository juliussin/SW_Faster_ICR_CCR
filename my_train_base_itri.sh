#!/bin/bash
save_dir="data/experiments/SW_Faster_ICR_CCR/itri/model"
dataset="itri"
pretrained_path="data/pretrained_model/resnet101_caffe.pth"  #vgg16-397923af.pth"  # resnet101_caffe.pth"
net="res101"
log_dir="data/logs/base_itri"

CUDA_LAUNCH_BLOCKING=1 python base_train_net.py --max_epochs 12 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --use_tensorboard --log_dir ${log_dir}

