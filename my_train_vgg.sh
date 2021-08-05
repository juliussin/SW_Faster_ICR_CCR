#!/bin/bash
save_dir="data/experiments/SW_Faster_ICR_CCR/cityscape/model"
dataset="cityscape"
pretrained_path="data/pretrained_model/vgg16_caffe.pth"  #vgg16-397923af.pth"
net="vgg16"
log_dir="data/logs/vgg16"

CUDA_LAUNCH_BLOCKING=1 python da_train_net.py --max_epochs 7 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --gc --lc #--use_tensorboard --log_dir ${log_dir} #--da_use_contex

