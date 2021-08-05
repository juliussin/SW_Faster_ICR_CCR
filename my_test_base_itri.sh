#!/bin/bash
net="res101"
part="nthu_Rome_test" # "itri_test" # 
model_path="data/experiments/SW_Faster_ICR_CCR/itri/model/itri_7.pth"
num_epoch="7"
output_dir="data/experiments/SW_Faster_ICR_CCR/itri/result_Rome"
dataset="itri"

python eval/test_base.py --cuda --vis\
    --part ${part} --net ${net} --dataset ${dataset} \
    --model_dir ${model_path} --output_dir ${output_dir} --num_epoch ${num_epoch}
