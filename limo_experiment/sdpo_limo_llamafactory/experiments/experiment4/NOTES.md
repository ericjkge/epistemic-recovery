# Experiment 4 — LoRA hyperparameter sweep

Goal: find the *minimum* LoRA configuration that recovers epistemic verbalization to
near-pretrained without overshooting. Experiments 1–2 used `rank=16, epochs=2` and
produced epistemic-token rates ~3–4× the SDPO baseline — likely style mimicry of
LIMO's long verification traces, not a clean capability recovery.

## Hypothesis

The current rank=16 / epochs=2 config:

- adapter capacity may exceed what's needed to undo the SDPO suppression mask
- training long enough at that capacity locks in LIMO's long verification style
- "epistemic alignment" (Figure 9 / `evaluate_epistemic_alignment.py` scalars)
  on rank=16/epochs=2 LoRA likely *exceeds* pretrained Qwen3-8B, which is the
  upper-bound target

A lower-rank, lower-epoch LoRA should:

1. land closer to pretrained Qwen3-8B on epistemic-alignment scalars
2. preserve most of the AIME pass@k gain
3. produce shorter responses with fewer trigram-loop pathologies

If `rank=2, epochs=0.5` under-shoots (still close to SDPO baseline) and
`rank=16, epochs=2` over-shoots (way past pretrained), the sweep should find
the saddle in between.

## Recovery framing

- **floor** = SDPO baseline (epistemic verbalization suppressed)
- **target** = pretrained Qwen3-8B (epistemic verbalization preserved)
- **anti-target** = a LoRA whose epistemic-alignment scalar is *further* past
  pretrained than pretrained is from baseline (style mimicry, not recovery)

Distance metric: per-trace mean log-prob of epistemic tokens on the LIMO probe
set, taken from the `scalars` field of probe-snapshot JSONs. We already log this
at every checkpoint via `eval/probe_during_training.py` — no new instrumentation.

For each config we compute `recovery_pct = (lora − sdpo) / (pretrained − sdpo)`.
The sweet spot is `recovery_pct ∈ [0.7, 1.1]` — most of the gap closed without
overshooting by more than 10%.

## Sweep grid

`configs.tsv` enumerates the cells. Initial grid:

| rank | epochs | lr   | rationale                                    |
|------|--------|------|----------------------------------------------|
| 2    | 0.5    | 5e-5 | smallest plausible — likely under-shoot      |
| 4    | 0.5    | 5e-5 |                                              |
| 4    | 1.0    | 5e-5 |                                              |
| 8    | 0.5    | 5e-5 |                                              |
| 8    | 1.0    | 5e-5 | first candidate "ideal"                      |
| 16   | 0.5    | 5e-5 | exp1/2 rank, fewer epochs                    |
| 16   | 1.0    | 5e-5 |                                              |
| 16   | 2.0    | 5e-5 | exp1/2 reproduction — anti-target            |

Sweep is sequential. Each cell: ~20–40 min train + ~5 min light eval on a
single H100. Whole grid ≈ 4–6 hours.

## What we measure per cell

The sweep ranks cells by alignment alone. Capability eval (AIME) is **deferred
to the top 1–2 winners** — running a full AIME pass on every cell during the
sweep wastes hours of H100 on tiebreak data the recovery_pct ranking doesn't
need.

### During the sweep (every cell)

1. **Epistemic alignment scalars** from `probe_during_training.py`'s snapshots:
   - `epistemic_mean_logprob` — primary distance metric (drives recovery_pct)
   - `all_tokens_mean_logprob` — drift on non-epistemic tokens (regression check)
   - `epistemic_alignment_gap` — secondary distance metric

These are **free** — `train.sh` already runs the probe in a background tmux
pane during training. We just have to read the JSON snapshots after the run.
No extra inference, no AIME pass per cell.

`sweep.sh` defaults to `SWEEP_SKIP_AIME=1` for this reason. Override with
`SWEEP_SKIP_AIME=0 bash sweep.sh` if you want full eval on every cell.

### After the sweep (winners only)

`analyze_sweep.py` writes `summary.csv` ranked by `|recovery_pct − 1|`.
Then `eval_winners.sh` runs full AIME (n=8, AIME24 + AIME25) on the top K:

```sh
bash eval_winners.sh                 # top 2
TOP_K=3 bash eval_winners.sh
CELLS=r8_e1_lr5e-5 bash eval_winners.sh   # explicit
```

This reuses parent `eval.sh`. Once it finishes, re-run `analyze_sweep.py` so
`summary.csv` and `recovery_vs_passk.png` pick up the new AIME numbers.

That gives us:

3. **AIME pass@1 / pass@8** on the top cells (~30 min/cell)
4. **Epistemic count + loop rate** from `analyze_epistemic.py` (falls out of the
   AIME eval JSONs)
5. **Avg response length** (proxy for "verification style mimicry")

Per-cell outputs land in `results/r{rank}_e{epochs}_lr{lr}/`.

## How to run

```sh
# Train every cell (sequential, resumable — skips cells whose adapter exists)
# AIME is skipped by default; only probe-derived alignment is computed.
bash sweep.sh

# Train a single cell
RANK=8 EPOCHS=1.0 LR=5e-5 bash sweep.sh

# After training the grid, rank cells by recovery_pct and write best_config.json:
python3 analyze_sweep.py --results_dir results --output summary.csv

# Then run full AIME on the top 1–2 winners (~30 min/cell at n=8):
bash eval_winners.sh

# Re-aggregate so summary.csv picks up the new AIME numbers:
python3 analyze_sweep.py --results_dir results --output summary.csv
```

`sweep.sh` is idempotent — re-running skips cells whose `adapter/adapter_model.safetensors`
already exists. Set `FORCE=1` to re-train everything. Set `SWEEP_SKIP_AIME=0`
if you want every cell evaluated on AIME during the sweep (slow).

## Pre-requisites

- `experiment2/eval_results/epistemic_probes/base_qwen3_8b_*.json` and
  `sdpo_no_lora_*.json` should already exist from experiment2. The aggregator
  reads them as the floor/target reference points. If absent, the first sweep
  cell will compute them (slow, ~10 min extra) and they'll be reused.
- Run from the parent `sdpo_limo_llamafactory/` directory's venv (`.venv-train`).

## Open questions for after the sweep

- Does the winning cell hold up at n=8 AIME and on AIME24 (OOD generalization)?
- Does it carry over to in-domain SciKnowEval (no regression)?
- Is the rank/epochs interaction monotonic, or does, e.g., `r=4, e=1` beat
  `r=8, e=0.5`? (i.e., is it about effective capacity-time, or independent
  effects?)
- Is `cutoff_len=8192` (truncating LIMO's right tail) a smaller-leverage way to
  achieve the same anti-mimicry effect? Worth a follow-up if rank/epochs alone
  doesn't suffice.
