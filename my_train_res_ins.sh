#!/bin/bash
save_dir="data/experiments/SW_Faster_ICR_CCR/cityscape/model"
dataset="cityscape"
pretrained_path="data/pretrained_model/resnet101_caffe.pth"  #vgg16-397923af.pth"  # resnet101_caffe.pth"
net="res101"
log_dir="data/logs/res101_ins"

CUDA_LAUNCH_BLOCKING=1 python da_train_net_new.py --max_epochs 12 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --gc --lc --use_tensorboard --log_dir ${log_dir} --da_use_contex

