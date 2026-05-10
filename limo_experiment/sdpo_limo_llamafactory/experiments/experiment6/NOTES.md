# Experiment 6 — Genuine-recovery LoRA SFT

## Goal

Train a LoRA on a curated SFT set composed entirely of **genuine recovery
traces** — chains where the SDPO base model committed to a wrong `\boxed{}`
answer in round 1 and then recovered to a correct answer in a later round
under the injection-prompt protocol from experiment 5. Test whether
~200 high-quality recovery rows are enough to elicit selective-verification
behavior on AIME25 without the dilution of LIMO's 813-trace mix.

## Dataset

Merged from two cohorts (`build_recovery_dataset.py`):

| Source | Rows | Notes |
|---|---:|---|
| Pass-4 cohort (`experiment5/outputs/pass4/injection_traces.jsonl`) | ~77 | R1-budget=20K, cumulative cap=30K. The strict committed-wrong→recovered set described in `experiment5/NOTES.md` §"Recovery dataset (R2-correct cohort)". |
| Legacy cohort (`experiment5/outputs/onpolicy_recovery_dataset.json`) | 126 | Earlier 18K-cap pass; already in alpaca form. |
| **Total** | **~203** | Above the ~150 stable-SFT floor referenced in `experiment5/NOTES.md` (heuristic, not derived). |

Per the spec, the two cohorts are **concatenated without dedupe** by default
— the priority is clearing the stability floor before worrying about
overlap. `DEDUPE=1 bash train.sh` enables question-level dedupe (pass-4
wins; legacy duplicates dropped).

Row format matches `make_limo_dataset.py` and `make_injection_dataset.py`:

```
{
  "instruction": <problem statement>,
  "input": "",
  "output": "<think>\n{full multi-round trace, injection markers preserved}\n</think>\n\nThe final answer is \\boxed{N}.",
  "system": "Please reason step by step, and put your final answer within \\boxed{}."
}
```

## Pipeline

```
build_recovery_dataset.py      → outputs/genuine_recovery_dataset.json (recovery mode)
build_limo_ablation_dataset.py → outputs/limo_ablation_dataset.json    (limo mode)
analyze_epistemic_tokens.py    → eval_results/sft_dataset_epistemic.txt
train.sh                       → adapter/ + merged/ via parent train.sh + dataset_info.json
eval.sh                        → eval_results/ via parent eval.sh, AIME25 only
```

## Usage

```bash
# Main run: ~204-row genuine recovery cohort
bash experiments/experiment6/train.sh

# Off-policy LIMO ablation: random 200 rows from limo_v2_sdpo.json,
# trained with the *same* hparams. Writes to experiments/experiment6_limo/.
MODE=limo bash experiments/experiment6/train.sh

# Match exact row count of the recovery cohort instead of fixing at 200
MODE=limo MATCH_N=1 bash experiments/experiment6/train.sh

# Build + analyze only (no training); applies to either mode
ANALYZE_ONLY=1 bash experiments/experiment6/train.sh
ANALYZE_ONLY=1 MODE=limo bash experiments/experiment6/train.sh

# AIME25 evaluation (point at whichever experiment dir to eval)
bash experiments/experiment6/eval.sh                          # main run
EXPERIMENT=experiment6_limo bash experiments/experiment6/eval.sh   # ablation
```

## LIMO off-policy ablation (MODE=limo)

Holds row count, hparams, prompt wrapper, and answer format constant; swaps
**trace source** from on-policy recovery traces (main exp6) to LIMO's
human-written solutions. This isolates the on-policy/off-policy axis.

| | recovery (main) | limo (ablation) |
|---|---|---|
| Source | exp5 pass-4 (78) + legacy (126) | random 200 from `limo_v2_sdpo.json` (seed=42) |
| Rows | ~204 | 200 (or matched via `MATCH_N=1`) |
| Trace origin | SDPO model's own retries | human-written |
| Output dir | `experiments/experiment6/` | `experiments/experiment6_limo/` |

Pre-training epistemic density (per 1K thinking words, from
`analyze_epistemic_tokens.py` on the SFT data):

| cohort | TOTAL | wait | maybe | alternatively |
|---|---:|---:|---:|---:|
| recovery (n=204) | 16.67 | 6.73 | 2.48 | 1.40 |
| limo (n=200)     | 36.72 | 10.44 | 6.90 | 6.65 |

LIMO traces are >2× more hedge-dense than the recovery cohort before any
training — consistent with the "over-verifies" failure mode previously seen
on the 813-row LIMO LoRA. The ablation lets us check whether that density
gap survives at the same row count, or whether it's a row-count effect.

**Hparams.** Mirror experiment 2's config (`rank=16 epochs=2 lr=5e-5`) with
the rank dropped to **8** since the cohort is ~4× smaller than LIMO's 813 —
fewer trainable LoRA params should match the smaller signal. Override any
field with `RANK= EPOCHS= LR=`.

## What we expect to see

The framing question is whether genuine-recovery training data produces a
model that **selectively** verifies on hard problems rather than over-verifies
across the board (the failure mode of the 813-row LIMO LoRA — see the
`science over-verification` finding in user/project memory).

Three measurable predictions to check against AIME25 results and the
epistemic-token table:

1. **AIME25 pass@4 ≥ exp4 winner.** If 203 rows are enough to elicit
   recovery, pass@k should not regress. If it does, the recovery cohort
   is too narrow to be useful as standalone SFT.
2. **Lower thinking-token mean than exp1/2 LoRA on AIME25.** The recovery
   cohort is shorter and more targeted than full LIMO traces; if the
   trained model inherits that, mean thinking tokens should drop relative
   to the 813-row LIMO LoRA on the same eval.
3. **Epistemic density on generations sits between SDPO baseline and exp1/2
   LoRA.** Concretely: count of `wait` / `but` / `actually` per 1K thinking
   words on AIME25 generations should be *higher* than baseline (otherwise
   the LoRA didn't pick up the recovery pattern at all) but *lower* than
   the 813-row LoRA (otherwise we just reproduced over-verification with
   less data).

If (1) regresses while (2)/(3) move in the predicted direction, the
conclusion is "recovery alone is too narrow — needs LIMO-style diversity
mixed in." That motivates an experiment 7 with a stratified mix.
