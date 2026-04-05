#!/bin/bash

nohup bash -c "
for waits in 1 2 4; do
    CUDA_VISIBLE_DEVICES=0 python eval.py \
        --model_name_or_path beanie00/math-SDPO-Qwen3-8B-think-step-100 \
        --data_path ../data/math/evaluation/aime24.parquet \
        --max_tokens 38912 \
        --enable_thinking \
        --temperature 0.6 \
        --top_p 0.95 \
        --n_sampling 16 \
        --k 16 \
        --wait_injections \${waits}
done
" > eval_wait_sdpo.log 2>&1 &
