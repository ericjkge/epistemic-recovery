#!/bin/bash

nohup bash -c "
for model in \
    Qwen/Qwen3-8B \
    beanie00/math-GRPO-Qwen3-8B-think-step-100 \
    beanie00/math-SDPO-Qwen3-8B-think-step-100
do
    CUDA_VISIBLE_DEVICES=0 python eval.py \
        --model_name_or_path \${model} \
        --data_path ../data/math/evaluation/aime24.parquet \
        --max_tokens 38912 \
        --enable_thinking \
        --temperature 0.6 \
        --top_p 0.95 \
        --n_sampling 16 \
        --k 16
done
" > eval_aime24_qwen3_8b.log 2>&1 &
