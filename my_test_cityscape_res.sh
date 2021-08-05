#!/bin/bash
net="res101"
part="test_t"
model_path="data/experiments/SW_Faster_ICR_CCR/cityscape/model/cityscape_7.pth"
num_epoch="7"
output_dir="data/experiments/SW_Faster_ICR_CCR/cityscape/result"
dataset="cityscape"

python eval/test_SW_ICR_CCR.py --cuda --gc --lc \
    --part ${part} --net ${net} --dataset ${dataset} \
    --model_dir ${model_path} --output_dir ${output_dir} --num_epoch ${num_epoch}