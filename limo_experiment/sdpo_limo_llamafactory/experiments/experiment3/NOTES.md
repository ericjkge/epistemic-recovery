# Experiment 3 — MI-peak comparison on AIME25 problem 23

Goal: test whether MI-peak density is a useful proxy for epistemic verbalization by
comparing matched traces on a problem where the LoRA succeeds and the SDPO baseline
fails. Method follows
[strategic-information-allocation-llm-reasoning/mi_peak/calc.py](../../../strategic-information-allocation-llm-reasoning/mi_peak/calc.py),
which is itself adapted from MI-Peaks (Qi et al. 2025, arXiv:2506.02867).

## Why problem 23

Per `experiment2/NOTES.md`, AIME25 problem 23 is one of three "LoRA-only solved"
problems on n=8 sampling: LoRA 4/8 correct, baseline SDPO 0/8. Question is the
`sin(7π·sin(5x)) = 0` tangency-counting problem with ground truth 149. Baseline
consistently produces 150 (off-by-one); LoRA's first correct sample is gen[2] = 149.

## Inputs

`prepare_inputs.py` pulls one matched pair from `experiment2/eval_results/`:

- `inputs/aime23_lora_correct.jsonl` — LoRA gen[2], extracted answer 149, ~29.7k chars
- `inputs/aime23_base_wrong.jsonl` — baseline SDPO gen[0], extracted answer 150, ~10.5k chars

Each JSONL is one record with `generated_responses` (the trajectory text) and
`gold_answer`, in the format `calc.py` expects.

## How to run

`run_mi_peak.sh` invokes calc.py twice. Defaults: layer 36 (matches `mi_peak/plot_mi.sh`
for Qwen3-8B), `max_tokens=4000`, GT mode = answer-only.

```sh
bash run_mi_peak.sh                 # uses defaults
LAYER=28 MAX_TOKENS=2500 bash run_mi_peak.sh
```

Outputs land in `results/{lora_correct,base_wrong}/mi_first{MAX_TOKENS}_layer{LAYER}_answer_only.csv`.

### Caveats

- **Merged LoRA weights are missing locally.** `experiment2/adapter/merged/` has
  config + tokenizer but no safetensor shards (~16GB). Re-merge with
  `eval/merge_lora.py` or set `LORA_MODEL` to a path that has full weights before
  running. Baseline pulls from HF.
- **GPU required.** Two forward passes on Qwen3-8B at 4k tokens each, plus per-token
  HSIC. CPU-only Mac is impractical.
- **Trace lengths differ** (LoRA ~30k chars vs baseline ~10k). Position-aligned
  comparison only meaningful within each trace; for cross-model comparison report
  peaks-per-1k-tokens, not raw counts.

## What we're measuring

Per-token MI-Peaks score = HSIC(layer-36 hidden state at token i, layer-36 last-token
hidden state of GT answer text), with σ=50. Peaks are local maxima above mean + k·std
of the same trace. The hypothesis: if MI peaks proxy epistemic verbalization, the
correct LoRA trace should show denser peaks aligned with verification/backtracking
tokens than the baseline's wrong trace.

## Followups (not yet run)

- Re-run with `--use_solution` to see if conditioning on the gold solution (not just
  the answer) sharpens peak structure.
- Sweep across layers — MI-Peaks paper finds peak structure varies by depth.
- Extend to all three LoRA-only-solved problems (006, 010, 023) before drawing any
  conclusions from a single example.
