#!/bin/bash
save_dir="data/experiments/SW_Faster_ICR_CCR/nthu_Tokyo/model"
dataset="nthu_Tokyo"
pretrained_path="data/pretrained_model/resnet101_caffe.pth"  #vgg16-397923af.pth"  # resnet101_caffe.pth"
net="res101"
log_dir="data/logs/nthu_Tokyo_oracle"

CUDA_LAUNCH_BLOCKING=1 python base_train_net.py --max_epochs 12 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --use_tensorboard --log_dir ${log_dir}

