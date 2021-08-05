#!/bin/bash
save_dir="data/experiments/SW_Faster_ICR_CCR/itri_nthu_Taipei/model"
dataset="ft_taipei"
pretrained_path="data/pretrained_model/resnet101_caffe.pth"  #vgg16-397923af.pth"  # resnet101_caffe.pth"
net="res101"
log_dir="data/logs/itri_nthu_Taipei_ft "
resume_name="itri_nthu_12.pth"

CUDA_LAUNCH_BLOCKING=1 python da_train_net_fine_tuning.py --max_epochs 15 --cuda \
  --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} \
  --gc --lc --use_tensorboard --log_dir ${log_dir} --da_use_contex \
  --resume True --resume_name ${resume_name} --lr 0.0005