# Experiment 7 — On-policy recovery LoRA at smaller effective batch

## Goal

Re-train the LoRA on experiment 6's ~204-row genuine-recovery cohort with
the **effective batch size halved** (`GRAD_ACCUM=4` vs the parent default
of `8`), and run a focused sanity sweep on 7 AIME25 problems to check
(a) baseline competence on early problems, and (b) whether the
recovery-only LoRA still hits the case-study problems where the original
LIMO LoRA succeeded.

## Dataset

Identical to experiment 6. Built by reusing
[`build_recovery_dataset.py`](../experiment6/build_recovery_dataset.py)
without modification:

| Source | Rows | Notes |
|---|---:|---|
| Pass-4 cohort (`experiment5/outputs/pass4/injection_traces.jsonl`) | 78 | R1-budget=20K, cumulative cap=30K. Strict committed-wrong → recovered set. |
| Legacy cohort (`experiment5/outputs/onpolicy_recovery_dataset.json`) | 126 | Earlier 18K-cap pass; already in alpaca form. |
| **Total** | **204** | Concatenated without dedupe (same as exp6 default). |

Output written to `outputs/genuine_recovery_dataset.json`; the
provenance-stripped training copy is registered as
`genuine_recovery_exp7` in `dataset_info.json`.

## Hparams

|  | exp6 | exp7 |
|---|---|---|
| rank | 8 | 8 |
| epochs | 2 | 2 |
| lr | 5e-5 | 5e-5 |
| `gradient_accumulation_steps` | 8 (parent default) | **4** |
| effective batch size | 8 | **4** |
| grad steps over 2 epochs | ~50 | **~100** |

Motivation. With ~204 rows and the parent's default batch=8, exp6 sees
only ~25 updates per epoch. Halving the batch doubles the update count
without changing the total tokens seen. The cohort is small enough that
the noisier per-step gradient is acceptable, and more updates may help
the LoRA actually converge on the recovery pattern rather than under-fit.

Override knobs:
```bash
GRAD_ACCUM=2 bash experiments/experiment7/train.sh   # ~200 grad steps, more aggressive
RANK=16 bash experiments/experiment7/train.sh        # match exp2's capacity
EPOCHS=3 bash experiments/experiment7/train.sh       # squeeze more updates
```

## Sanity eval

[`sanity_eval.py`](sanity_eval.py) runs vLLM on a 7-problem subset of
AIME25 with `n_sampling=4` for both `baseline` (SDPO) and `lora`
(SDPO + exp7 adapter):

| problem_id | role | ground truth |
|---|---|---:|
| `aime25_000` .. `aime25_004` | easy probes (problems 1–5) | varies |
| `aime25_017` | 2×2 grid 2-red/2-blue colorings (Case 2) | 82 |
| `aime25_023` | sin(7π sin 5x) zeros + tangents (Case 1) | 149 |

The case studies come from
[`experiments/experiment1/eval_results/epistemic_case_studies.md`](../experiment1/eval_results/epistemic_case_studies.md).
On both, the original 813-row LIMO LoRA recovered to the correct answer
where the SDPO baseline collapsed to a clean-but-wrong shortcut
(150 vs 149 endpoint exclusion; 16/8/12 vs 82 multiplicity miscount).

Cost: 7 problems × 4 samples × 2 models ≈ 56 generations. Few minutes
on a single H100.

## What we expect to see

Three readings, in order of importance:

1. **Easy probes (`aime25_000`..`aime25_004`) acc ≥ baseline.** This
   is the bare correctness gate — if the recovery LoRA breaks problems
   the SDPO baseline already solves, we've over-fit to the recovery
   pattern and need more diversity.
2. **Case-study lift on `aime25_017` / `aime25_023`.** The baseline
   gets 0/4 on both per the experiment 1 trace. A single correct sample
   on either is meaningful. Matching the LIMO LoRA's behavior (≥ 2/4 on
   either case) would be strong evidence the recovery cohort transfers.
3. **Token economy.** The recovery cohort is shorter and more targeted
   than full LIMO traces, so mean thinking-token length on these 7
   problems should not blow up vs baseline. If it does, the LoRA is
   inheriting over-verification despite the trace cohort being shorter.

If (1) holds and (2) shows any lift, exp7 is a yes-go for a fuller
AIME25 sweep — re-run with the parent eval pipeline:
```bash
EXPERIMENT=experiment7 bash eval.sh   # parent eval.sh, full AIME24+25
```
If (1) regresses, halving the batch isn't the missing piece; the next
move is likely a stratified mix with LIMO traces (exp8).

## Pipeline

```
build_recovery_dataset.py  → outputs/genuine_recovery_dataset.json   (reused from exp6)
train.sh                   → adapter/ + merged/ via parent train.sh
sanity_eval.py             → eval_results/{baseline,lora_..}_aime25.json (7 problems)
eval.sh                    → wraps sanity_eval.py
```

## Usage

```bash
# Train (reuses exp6's recovery dataset under name `genuine_recovery_exp7`)
bash experiments/experiment7/train.sh

# Knobs
GRAD_ACCUM=2 bash experiments/experiment7/train.sh    # more aggressive batch
SKIP_DATASET=1 bash experiments/experiment7/train.sh  # reuse existing dataset

# Sanity eval (7-problem AIME25 subset)
bash experiments/experiment7/eval.sh
N_SAMPLING=8 bash experiments/experiment7/eval.sh
MODELS=baseline,lora,pretrained bash experiments/experiment7/eval.sh
PROBLEM_IDS=aime25_023 bash experiments/experiment7/eval.sh   # one-problem probe
```
