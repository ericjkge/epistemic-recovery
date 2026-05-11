# Experiment 8 — Hybrid recovery + LIMO LoRA SFT (narrativized, hygiene-first)

## Goal

Train a LoRA SFT on a hybrid of (a) on-policy injection-recovery rows and
(b) off-policy LIMO rows under **strict data hygiene**, hoping to combine
LIMO's benchmark coverage with the recovery cohort's hedge-and-retry
behavior — *without* inheriting LIMO's "verify-forever" tail or producing
a `**Final Answer** \boxed{X}` → "Wait" → `**Final Answer** \boxed{Y}` →
"Wait" mode-collapse cycle that a naive hybrid would teach.

The hygiene rule is the load-bearing piece. Both source cohorts violate it
out of the box:

- **LIMO-v2 traces.** Human authors use `\boxed{intermediate}` as inline
  LaTeX throughout solutions; `make_limo_dataset.py` only strips the
  *last* `\boxed{}`, leaving earlier ones inside `<think>`.
- **Pass-4 on-policy recovery traces.** The SDPO base model emits
  `**Final Answer**\n\boxed{X}` blocks inside thinking by habit. The
  injection protocol's `reopen_thinking()` strips post-`</think>` content
  but leaves the pre-`</think>` "Final Answer" block in place.

Trained naively on this data, the model learns `\boxed{}` and
`**Final Answer**` are routine emphasis markers rather than terminal
commits — and at inference, "Wait, that might be wrong" between two of
those markers becomes a stable mode-collapse loop.

This experiment enforces one global rule on every training row:

> **`\boxed{}` and `**Final Answer**` appear EXACTLY ONCE per response, in
> the terminal commit AFTER `</think>`. Inside `<think>...</think>`, the
> trace is pure prose — no boxed values, no Final-Answer markers.**

## Pipeline

```
narrativize.py                    → in-memory transform: strip \boxed{}/markers from <think>
build_clean_recovery_dataset.py   → exp 8 strict-clean 56 rows (one-off; output committed)
build_hybrid_narrative_dataset.py → outputs/hybrid_narrative_dataset.json
                                     (+ outputs/hybrid_narrative_dataset_train.json
                                      registered as hybrid_narrative_exp8)
train.sh                          → adapter/ + merged/ via parent train.sh
sanity_eval.py / eval.sh          → eval_results/ (7-problem AIME25 subset)
```

The on-disk dataset at [`outputs/hybrid_narrative_dataset.json`](outputs/hybrid_narrative_dataset.json)
is the result of the hygiene pass. `train.sh` defaults to `SKIP_DATASET=1`
so it reuses that file; set `SKIP_DATASET=0` to rebuild (which requires
`outputs/clean_recovery_dataset.json` or an override `RECOVERY_CLEAN`
pointing at e.g. `experiments/experiment6/outputs/genuine_recovery_dataset.json`).

## Dataset

| Cohort | Source | Rows | Transform |
|---|---|---:|---|
| On-policy recovery | exp 8 `clean_recovery_dataset.json` (56 strict-clean) | **56** | narrativized |
| Off-policy LIMO | top-600 by density under 75th-pct length cap from `limo_v2_sdpo.json` | **600** | narrativized |
| **Total** | | **656** | |

### Narrativization transform (applied to every row's `output`)

The transform operates **only inside the `<think>...</think>` span**. The post-`</think>`
tail (the single terminal commit) is preserved verbatim.

1. **Remove `**Final Answer**\n\boxed{...}` commit blocks**, with brace-balanced
   matching for nested LaTeX (`\boxed{\frac{a}{b}}`, `\boxed{\sqrt{3}}`, etc.).
   Also matches the markdown-heading variant (`### Final Answer\n\boxed{...}`)
   and the math-display variant (`### Final Answer\n$$\n675\n$$`).
2. **Replace any remaining inline `\boxed{X}` with `X`** (brace-balanced).
   Empty `\boxed{}` is preserved — the model often quotes the system prompt's
   "put your final answer within `\boxed{}`" verbatim, which is legitimate prose.
3. **Remove leftover `Final Answer` markers** in any markdown form
   (`**Final Answer**`, `### Final Answer`, `Final Answer:`, plain `Final Answer`).
4. **Collapse runs of >2 blank lines to 2.**

Pre-transform vs post-transform stats (verified against the on-disk dataset):

| | recovery 56 | LIMO 600 |
|---|---:|---:|
| `\boxed{X}` inside `<think>` (before) | 151 | 2,122 |
| `\boxed{X}` inside `<think>` (after) | **0** | **0** |
| "Final Answer" inside `<think>` (before) | 114 | 780 |
| "Final Answer" inside `<think>` (after) | **0** | **0** |
| post-`</think>` unchanged | 56/56 | 600/600 |

### Why the recovery cohort is small (56 rows)

From the 78 pass-4 genuine recoveries and 126 legacy 18K-cap recoveries
(experiment 6's merged cohort, 204 rows total), only 56 pass the strict
"clean recovery" filter:

- R1 committed to a wrong `\boxed{}` before injection
- Exactly one injection-phrase occurrence
- R2 didn't oscillate (every R2 box, if any, equals the ground-truth final)
- At most one "Final Answer" marker in R2
- Legacy rows had their old phrase ("Wait, the answer is wrong. Let's think
  again.") rewritten to the current phrase ("Wait, that might be wrong.
  Let's try some different approaches.") before filtering.

The other ~148 rows had one or more pathologies — multi-injection loops,
R2 quiet revision, R2 mode-collapse stamping, missing R1 commit.

The cohort is below the ~150-row stable-SFT floor mentioned in
`experiment5/NOTES.md`. **Growing this pool is the next experimental
move** — see the section below.

## Hparams

|  | exp 7 | **exp 8** | rationale |
|---|---|---|---|
| rank | 8 | **16** | match exp 2's capacity (LIMO slice doubles the data; rank doubles too) |
| epochs | 2 | **2** | identical |
| lr | 5e-5 | **5e-5** | identical |
| `grad_accum` | 4 | **8** | parent default; ~656 rows tolerates the larger effective batch |
| cutoff_len | 32,768 | **32,768** | identical |
| dataset size | 204 | **656 (56 + 600)** | hybrid: recovery cohort + LIMO 600 |

Override knobs: `RANK= EPOCHS= LR= GRAD_ACCUM= CUTOFF_LEN=`.

## Pipeline usage

```bash
# End-to-end: train (reuses on-disk dataset) + eval
cd /home/ubuntu/epistemic-recovery/limo_experiment/sdpo_limo_llamafactory
bash experiments/experiment8/train.sh && bash experiments/experiment8/eval.sh

# Force a dataset rebuild (needs outputs/clean_recovery_dataset.json or override)
SKIP_DATASET=0 bash experiments/experiment8/train.sh

# Reuse adapter, just rerun sanity eval
bash experiments/experiment8/eval.sh
N_SAMPLING=8 bash experiments/experiment8/eval.sh
PROBLEM_IDS=aime25_023 bash experiments/experiment8/eval.sh
```

## What we expect to see

The hypothesis is narrow: **hygiene alone (no other changes) should kill the
mode-collapse loop pathology that a naive hybridization would produce.**
Three measurable predictions:

1. **No `\boxed{}` stamping at inference.** Count `\boxed{}` in the
   `<think>` span of LoRA generations. A naive hybrid LoRA can emit
   1,100+ mid-trace boxed values on the hardest problems. Exp 8's LoRA
   should emit zero (or near-zero — any emission is a sign the LoRA
   didn't fully absorb the hygiene rule from 656 rows).

2. **AIME25 sanity ≥ exp 7 baseline probes.** The 7-problem subset
   (5 easy probes + `aime25_017` + `aime25_023`). At minimum, the easy
   probes should not regress: if the LoRA is properly committing
   `\boxed{}` exactly once at the end, accuracy on the 5 easy probes
   should match or beat baseline. Case-study lift is harder; we'd accept
   ≥ 1/4 on either.

3. **Thinking token economy.** A pathological LoRA's worst-case
   generations can burn 30K tokens on stamping; exp 8 should not. Median
   `<think>` token count on the 7 sanity problems should sit between
   baseline (~6K) and the 813-row LIMO LoRA's ~12K — not orders of
   magnitude higher.

If (1) holds and (2) doesn't regress, hygiene is confirmed as the right
fix and the next step is growing the clean recovery cohort (next section).

## Growing the clean recovery cohort — one more LIMO injection pass

The 56-row strict-clean cohort sits below the ~150-row stable-SFT floor.
Now that the data hygiene pipeline (`narrativize.py` +
`build_hybrid_narrative_dataset.py`) is in place and validated, the
follow-up move is to run **one more pass of `generate_injections.py`** on
the same diagnostic seed set with a fresh sampling seed, combine into a
pass-5 cohort, and re-filter.

Pass-1 used `seed=42`, pass-3 used `seed=43`, so pass-5 below uses
`seed=44`. The diagnostic seed set drops the 167 LIMO-v2 problems where
all 4 chains R1-correct in pass-1 (those can't contribute recoveries).

```bash
# ── 0. Setup ────────────────────────────────────────────────────────────────
cd /home/ubuntu/epistemic-recovery/limo_experiment/sdpo_limo_llamafactory
EXP5=experiments/experiment5
EXP6=experiments/experiment6
EXP8=experiments/experiment8

# ── 1. New on-policy injection pass (seed=44, single chain, ~80 min on 1×H100)
mkdir -p "$EXP5/outputs/pass5"
setsid nohup python3 "$EXP5/generate_injections.py" \
    --seeds "$EXP5/outputs/diagnostic_seeds.jsonl" \
    --output_dir "$EXP5/outputs/pass5" \
    --num_chains 1 \
    --max_rounds 2 \
    --max_tokens_r1 20000 \
    --max_tokens_r2 10720 \
    --model baseline \
    --seed 44 \
    < /dev/null > "$EXP5/outputs/pass5.log" 2>&1 &
disown

# Monitor:
#   tail -f $EXP5/outputs/pass5.log
#   nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

# ── 2. Combine with the existing pass-4 cohort → pass-5 (5 chains/problem) ──
python3 "$EXP5/combine_passes.py" \
    --inputs "$EXP5/outputs/pass4/injection_traces.jsonl" \
             "$EXP5/outputs/pass5/injection_traces.jsonl" \
    --output_dir "$EXP5/outputs/pass5_combined"

# ── 3. Rebuild the genuine-recovery alpaca cohort from the merged traces ────
python3 "$EXP6/build_recovery_dataset.py" \
    --pass4_traces "$EXP5/outputs/pass5_combined/injection_traces.jsonl" \
    --output "$EXP6/outputs/genuine_recovery_dataset_pass5.json"

# ── 4. (Optional) Strict-clean filter; if you keep the inline filter logic
#       elsewhere, run it now to produce outputs/clean_recovery_dataset.json.
#       Otherwise, point build_hybrid_narrative_dataset.py at the genuine cohort
#       directly — narrativize.py still strips mid-trace boxes/markers.

# ── 5. Rebuild the hybrid SFT dataset (narrativize both cohorts + merge) ────
python3 "$EXP8/build_hybrid_narrative_dataset.py" \
    --recovery_clean "$EXP6/outputs/genuine_recovery_dataset_pass5.json" \
    --limo_dataset limo_v2_sdpo.json \
    --n_limo 600 \
    --length_pct_cap 75 \
    --seed 42 \
    --output "$EXP8/outputs/hybrid_narrative_dataset_pass5.json" \
    --strict

# ── 6. Train + eval against the larger cohort ────────────────────────────────
DATASET_PATH="$PWD/$EXP8/outputs/hybrid_narrative_dataset_pass5.json" \
DATASET_NAME=hybrid_narrative_exp8_pass5 \
EXPERIMENT=experiment8_pass5 \
SKIP_DATASET=1 \
    bash "$EXP8/train.sh"

EXPERIMENT=experiment8_pass5 bash "$EXP8/eval.sh"
```

What to read off the pass-5 run:

- **Pass-5 R1→R2 lift** (in `outputs/pass5/stats.json`). Should match
  pass-4's ~3.7pp; if substantially lower, the diagnostic seeds are
  saturated and additional chains are diminishing returns.
- **Genuine recovery count** after `build_recovery_dataset.py`. Pass-4
  produced 78 genuine recoveries; pass-5 should add ~20-40 net new ones
  (the pass-1+pass-3 chains already covered the easy recoveries).
- **Strict-clean yield**. The 78 → 56 strict-clean ratio (~72%) is the
  baseline; expect similar on net-new pass-5 rows. Target: ≥ 100 rows
  total post-filter, which crosses the SFT-floor.

## Caveats

- **Cohort imbalance.** 56 on-policy : 600 off-policy is a 1:11 ratio. Most
  gradient steps will be LIMO-driven. The model could land closer to a pure
  LIMO LoRA than a true hybrid. Compare against exp 2's full-LIMO LoRA to
  see if the small recovery slice contributes anything observable.

- **Filter false-rejects.** The strict filter excluded 148/204 rows. Some
  of those rejected rows were genuine recoveries that just tripped a
  structural check (e.g., R2 happened to box the right answer once mid-trace
  before settling). An LLM judge could probably recover ~10-15 of these.
  Not pursued here to keep the hygiene change isolated.

- **Empty `\boxed{}` quirk.** The transform preserves empty `\boxed{}` because
  the model quotes the system prompt verbatim ("put your final answer within
  \boxed{}"). The trace still contains the literal string `\boxed{}` in a
  prose context. If this becomes a problem (model learns to emit empty
  `\boxed{}` at inference), tighten the transform to wrap the literal
  reference in code formatting or replace with a placeholder.

- **No regeneration was needed for the on-disk dataset.** All hygiene fixes
  are post-hoc on existing data. The pass-4 injection_traces.jsonl is
  unchanged; we only re-filtered and re-formatted into a training-ready
  dataset. The pass-5 command above is what the next iteration looks like.

## Lineage

```
experiments/experiment5/outputs/pass4/injection_traces.jsonl
  ↓ build_recovery_dataset.py  (exp 6)
experiments/experiment6/outputs/genuine_recovery_dataset.json  (204 rows)
  ↓ strict-clean filter  (one-off; legacy phrase rewrite + 5 structural checks)
experiments/experiment8/outputs/clean_recovery_dataset.json    (56 rows)
  ↓ build_hybrid_narrative_dataset.py  (narrativize + merge with LIMO 600)
experiments/experiment8/outputs/hybrid_narrative_dataset.json  (656 rows)
  ↓ train.sh
experiments/experiment8/adapter/  + adapter/merged/
  ↓ eval.sh
experiments/experiment8/eval_results/
```
