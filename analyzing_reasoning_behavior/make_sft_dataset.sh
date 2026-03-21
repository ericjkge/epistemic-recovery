CUDA_VISIBLE_DEVICES=4,5,6,7 python make_sft_dataset.py \
    --input ./outputs_compare/comparison_results.jsonl \
    --output_dir ./sft_datasets