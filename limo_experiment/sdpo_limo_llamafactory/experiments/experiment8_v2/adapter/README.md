---
library_name: peft
license: other
base_model: beanie00/math-SDPO-Qwen3-8B-think-step-100
tags:
- base_model:adapter:beanie00/math-SDPO-Qwen3-8B-think-step-100
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: adapter
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# adapter

This model is a fine-tuned version of [beanie00/math-SDPO-Qwen3-8B-think-step-100](https://huggingface.co/beanie00/math-SDPO-Qwen3-8B-think-step-100) on the hybrid_narrative_exp8_v2 dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4777

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
- num_epochs: 2

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.4991        | 1.0   | 87   | 0.4824          |
| 0.4556        | 2.0   | 174  | 0.4777          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.6.0
- Pytorch 2.5.1+cu121
- Datasets 4.0.0
- Tokenizers 0.22.2