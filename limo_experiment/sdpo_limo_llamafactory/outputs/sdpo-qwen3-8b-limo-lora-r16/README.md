---
library_name: peft
license: other
base_model: beanie00/math-SDPO-Qwen3-8B-think-off-step-100
tags:
- base_model:adapter:beanie00/math-SDPO-Qwen3-8B-think-off-step-100
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: sdpo-qwen3-8b-limo-lora-r16
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sdpo-qwen3-8b-limo-lora-r16

This model is a fine-tuned version of [beanie00/math-SDPO-Qwen3-8B-think-off-step-100](https://huggingface.co/beanie00/math-SDPO-Qwen3-8B-think-off-step-100) on the limo_v2_sdpo dataset.
It achieves the following results on the evaluation set:
- Loss: 0.5102

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.05
- num_epochs: 2.0

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.4757        | 1.0   | 95   | 0.5146          |
| 0.4941        | 2.0   | 190  | 0.5102          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.2.0
- Pytorch 2.11.0+cu130
- Datasets 4.0.0
- Tokenizers 0.22.2