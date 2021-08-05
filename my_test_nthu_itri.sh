#!/bin/bash
net="res101"
part="nthu_Rome_test" #"itri_test" #
model_path="data/experiments/SW_Faster_ICR_CCR/itri_nthu_Rome/model/itri_nthu_8.pth"
num_epoch="8"
output_dir="data/experiments/SW_Faster_ICR_CCR/itri_nthu_Rome/result_Rome"
dataset="itri_nthu"

python eval/test_SW_ICR_CCR.py --gc --lc --cuda --vis\
    --part ${part} --net ${net} --dataset ${dataset} \
    --model_dir ${model_path} --output_dir ${output_dir} --num_epoch ${num_epoch}
