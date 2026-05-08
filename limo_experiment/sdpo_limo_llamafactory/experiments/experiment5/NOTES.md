# Experiment 5 — On-policy injection-recovery training

Goal: build an on-policy SFT dataset by letting a model attempt competition-math
problems, then injecting "Wait, the answer is wrong. Let's think again." at the
end of any wrong attempt and resampling — s1-style budget forcing applied to
*correctness* rather than length. Train a LoRA on the resulting traces and check
whether it beats the off-policy LIMO LoRA (experiments 1–4).

**Depends on Experiment 4.** The training step uses experiment 4's best
`(rank, epochs, lr)` config; the generation step optionally uses experiment 4's
best LoRA as the policy. Run experiment 4 first.

## Why this might work

Two well-tested ingredients combined:

1. **s1-style injection (budget forcing).** Replacing `</think>` with a
   continuation prompt extends reasoning and reliably lifts pass@1 on benchmarks
   the base model nearly solves. Muennighoff et al. (2025) get a few-percentage-
   points lift from a single "Wait" injection.
2. **On-policy SFT.** Training on traces drawn from your own policy avoids the
   distribution mismatch that off-policy LIMO traces introduce. The exp1/2 LoRA
   inherits LIMO's "verify forever" tail length and trigram-loop pathology;
   on-policy traces should not.

Combining them: use injection at *generation* time to manufacture multi-attempt
traces where the model recovers from its own errors. Train on the corrected
traces. The LoRA learns "when uncertain, verify and retry," not LIMO's
"verify-forever" style.

The **expected signal** during generation: a measurable bump from round-1
pass@1 to round-N pass@1 (≈ s1-style gain). If we don't see that lift on the
training-time generations, something's wrong with the injection setup before we
spend training compute.

## Pipeline

```
1. prepare_seed_problems.py   → seed_problems.jsonl
2. generate_injections.py     → outputs/injection_traces.jsonl + stats.json
3. make_injection_dataset.py  → outputs/onpolicy_injection_dataset.json (alpaca)
4. train.sh                   → adapter/ + eval_results/ via parent train.sh + eval.sh
```

### Step 1 — Seed problems

**Default source: GAIR/LIMO-v2** — the same 813 competition-math questions the
off-policy LIMO LoRA (experiments 1, 2, 4) was trained on. Reusing this exact
question set is deliberate:

1. **Clean ablation.** Experiments 1–4 train on LIMO's *human-written solutions*.
   Experiment 5 trains on *the model's own traces* on the same questions. Same
   prompts, different output distribution — so any difference in downstream
   AIME pass rate is attributable to the on-policy / off-policy axis, not the
   question distribution.
2. **Right-sized.** LIMO is 813 problems → ~70–80% have integer answers
   (filterable with the same `int()` grader as `evaluate_aime.py`) → ~550–650
   usable seeds, well above the ~150-problem floor for stable SFT.
3. **No eval contamination.** AIME24/25 are post-2023; LIMO predates both.

Override with `--source` to use any HF `{problem, answer}` dataset or a local
JSONL — e.g., `AI-MO/aimo-validation-aime` (90 pre-2024 AIME problems) for a
smaller, fully held-out seed set if you want to test transfer separately from
the on/off-policy axis.

### Step 2 — Iterative injection generation

For each seed problem, sample one response (`temperature=0.6`, `top_p=0.95`,
matching `evaluate_aime.py`). Extract the last `\boxed{...}`, grade against GT.

- **If correct:** record the trace as a positive example, done.
- **If wrong:** strip everything from the last `</think>` onward, append the
  injection phrase as a continuation of `<think>` content, resample.
  Repeat up to `max_rounds` (default 3) or until correct or until the
  cumulative output hits a token cap.

Each problem ends up with `{rounds, final_correct, final_trace}` where
`final_trace` is the concatenation of all rounds (with injection breaks visible
as in-trace markers — the model literally produced "Wait, the answer is wrong.
Let's think again." each retry).

Output: `outputs/injection_traces.jsonl` with one record per seed problem.
`outputs/stats.json` reports round-by-round pass rate — this is the s1-analog
sanity check.

### Step 3 — Dataset construction

Keep only problems where the *final* round was correct. Wrap as
LlamaFactory alpaca format (matching `make_limo_dataset.py`):

```json
{
  "instruction": "{question}\n\nPlease reason step by step, ...",
  "input": "",
  "output": "<think>\n{combined_thinking_with_injections}\n</think>\n\nThe final answer is \\boxed{{N}}.",
  "system": "Please reason step by step, ..."
}
```

The combined thinking is exactly what was sampled — *including* the injection
phrase between rounds. The trained LoRA sees those phrases as part of its own
distribution and learns to emit them when uncertain at inference time.

### Step 4 — Training

Use the winning hyperparameters from experiment 4. `train.sh` reads
`../experiment4/best_config.json` if present (written by `analyze_sweep.py` or
manually) and falls back to env-var overrides:

```sh
# After experiment 4 completes:
python3 ../experiment4/analyze_sweep.py --results_dir ../experiment4/results
# Edit best_config.json by hand if you disagree with the auto-pick, then:
bash train.sh

# Or override directly:
RANK=8 EPOCHS=1.0 LR=5e-5 bash train.sh
```

`train.sh` registers our dataset in a temporary `dataset_info.json` entry
(`onpolicy_injection`) and points the parent `train.sh` at it via the
`DATASET` env var (we add support for that in this experiment's wrapper).

## What we measure

1. **Round-1 → Round-N pass rate during generation.** If injection doesn't
   meaningfully raise correctness on the seed set, the dataset is just exp1/2
   re-skinned. We want >5pp lift from R1 to R3 on AIME-pre-2024 seeds.
2. **AIME24 + AIME25 pass@k after training.** Compare against:
   - SDPO baseline
   - exp4 winner (off-policy LIMO LoRA, low rank)
   - this experiment (on-policy injection LoRA, same hparams as exp4 winner)
3. **Epistemic alignment scalars.** Same probe pipeline as exp4. Should land in
   the same `[0.7, 1.1]` recovery band — if it overshoots like exp1/2, the
   injection traces are still LIMO-style-long.
4. **Loop rate + response length.** From `analyze_epistemic.py`. Hypothesis:
   on-policy traces have less degenerate repetition than LIMO-trained.

## Caveats / open questions

- **Choice of generation policy.** Defaults to `POLICY=lora` (exp4 best LoRA)
  for richer base-rate epistemic verbalization. `POLICY=baseline` (raw SDPO)
  is the strict on-policy choice — we'd be training the next iteration of
  baseline-SDPO from baseline-SDPO's own generations. Compute permitting, run
  both and compare.
- **Memorization caveat with `POLICY=lora` + LIMO seeds.** The exp4 LoRA was
  trained on the same questions we're now generating from. Round-1 pass rate
  will be inflated relative to novel problems, and the *injection lift*
  (R1 → R3) will be smaller because there are fewer wrong answers to retry.
  For the cleanest "does injection help" signal, run `POLICY=baseline` on
  LIMO seeds first. Then `POLICY=lora` for the on-policy training data.
- **Seed set size vs pass@k.** 813 LIMO seeds × ~70% integer-answerable ×
  3 rounds at ~12K tokens each ≈ 6–8 H100-hours per generation pass. Use
  `SEED_MAX=200` to cap during pipeline shake-out.
- **Injection phrase wording.** Default: `" Wait, the answer is wrong. Let's
  think again.\n\n"`. s1 just used `"Wait"`. The longer phrase is unambiguous
  but signals "this was graded" more strongly than s1's open-ended hedge —
  worth ablating once the pipeline works.
- **Cap on cumulative tokens.** We cap at 30,720 cumulative generated tokens
  (3 rounds × 10,240 per-round `max_tokens`) in `generate_injections.py`,
  matching `cutoff_len=32768` in `qwen3_sdpo_lora_sft.yaml` (Qwen3-8B's native
  context). This sizes the trace distribution to LIMO-v2's p99 (~31K tokens) so
  the verbose, epistemically-marked tail isn't systematically clipped — the
  whole point of experiment 5 is to match LIMO's reasoning length.
  `cumulative_token_cap` counts only generated tokens (excludes the prompt and
  the ~13-token injection phrases, which feed into the next round's prompt
  rather than being generated). Round 3 worst-case vLLM context use is ~31K,
  inside `max_model_len=32768`.
- **Risk of "self-fulfilling" injection.** If the model learns that
  "Wait, the answer is wrong. ..." always precedes a correct answer in
  training data, it may emit the phrase liberally at inference and skip
  honest verification. Worth checking final-token distributions post-training.

## Results — generation pass 1 (baseline SDPO policy, 10,240 per-round cap)

Run config: `model=baseline`, `num_chains=4`, `max_rounds=3`, `max_tokens=10240`,
`cumulative_token_cap=30720`, 795 LIMO-v2 seeds. See `outputs/stats.json`.

### Headline pass-rate lift (sanity check)

| Round | pass@1 (per chain) | pass@4 (per problem) |
|------:|-------------------:|---------------------:|
| 1     | 41.7% (1325/3180)  | 64.4% (512/795)      |
| 2     | 50.2% (1596/3180)  | 72.2% (574/795)      |
| 3     | 51.9% (1649/3180)  | 74.3% (591/795)      |

R1→R3 lift = **+10.2 pp pass@1, +9.9 pp pass@4** — passes the >5pp threshold.
Most of the lift is R1→R2; R2→R3 only adds ~1.7pp.

324 of 3180 chains (10.2%) flip wrong→correct after at least one injection.

### SFT dataset built (`outputs/onpolicy_injection_dataset.json`)

1,649 rows total — one row per *correct chain*:
- 1,325 R1-correct (no injection in trace)
- 271 recovered at R2 (one injection)
- 53 recovered at R3 (two injections)

The 1,531 dropped chains never got correct.

### Token-cap contamination — important

**21% of R1 attempts hit the 10,240 per-round cap; 16% (503/3180) hit the cap
*and* emitted no `\boxed{}` answer**, so they were graded "wrong" purely due
to truncation. **219 of the 324 injection-recoveries (~68%) had a truncated
R1** — meaning a chunk of the headline lift is a *budget* effect (R2/R3
gives the model another 10K tokens to keep going), not an *injection* effect.

Restricting to chains where R1 actually committed a wrong boxed answer:
- ~1,313 chains had committed-wrong R1
- ~105 of those genuinely recovered → **~8% true epistemic recovery rate**
  (vs. the 10.2% headline)

Baseline SDPO reasoning length context: median R1 = 5.8K tokens, p90 ≈ 9.6K
(against the cap, so true p90 is higher). Baseline on AIME averages 7.7K
(AIME24) to 8.2K (AIME25) thinking tokens. So the 10,240 cap is generous
for typical responses but bites on the harder ~20%.

## Decision — generation pass 2 config

Move to **`max_rounds=2`, `max_tokens=15000-16000`,
`cumulative_token_cap=29000`** to:

1. Cut R1 truncation from 21% → ~2-3%, eliminating the budget-vs-injection
   confound on the recovery measurement.
2. Stay safely inside Qwen3-8B's 32K native context. (Three rounds at 16K
   each is impossible inside 32K — R2's prompt already includes R1's full
   trace.)

R3 contributes only 1.7% recovery rate (53/3180 chains, 46 in the
recovery-only filter), so dropping `max_rounds=3` costs little.

**Existing R3 traces are kept** — they're already on disk in
`onpolicy_injection_dataset.json` and `onpolicy_recovery_dataset.json`. The
re-run only adds new data; nothing lost.

## Genuine-recovery filter — `outputs/onpolicy_recovery_dataset.json`

A stricter version of the SFT dataset that keeps only chains where the
model **committed a wrong boxed answer and self-corrected**:

- Case A: R1 emitted a wrong boxed answer; R2 correct
- Case B: R2 emitted a wrong boxed answer; R3 correct (R1 may be truncated)

| Case | Definition | Rows |
|---|---|---:|
| A | wrong R1 → correct R2 | 80 |
| B | wrong R2 → correct R3 | 46 |
| **Total** | | **126** |

Of the 46 Case B rows, **29 are "strict"** — both R1 and R2 committed wrong
boxed answers (two genuine self-corrections in one trace).

Caveat: 126 rows is below the ~150-problem floor for stable LoRA SFT noted
above. Either grow the dataset via the pass-2 re-run, or mix with a sample
of R1-correct rows from `onpolicy_injection_dataset.json` for stability.

Worked examples in `outputs/recovery_cases.md` (5 illustrative traces:
A.1 seating, A.2 Rubik's, A.3 power tower; B.1 parallelogram, B.2 cube
plane). Recovery types observed: constraint misreading, last-step
inversion, structural rethink, hypothesis search, criterion bug.

## Pass-2 re-run targets — `outputs/rerun_candidates.jsonl`

513 of 795 problems flagged for re-run, prioritized:

| Bucket | Count | Rerun? | Reason |
|---|---:|:---:|---|
| Already have ≥1 genuine recovery | 115 | skip | 126 traces secured |
| All 4 chains R1-correct (too easy) | 167 | skip | no recovery possible |
| **Has ≥1 truncated R1** | **226** | **high** | best yield from larger cap |
| Committed wrong, never recovered | 144 | medium | re-roll for stochastic recoveries |
| All chains failed entirely | 143 | low | hard problems; may still fail |

Outputs:
- `rerun_candidates.jsonl` — per-problem record with priority + chain stats
- `rerun_problem_ids.txt` — flat ID list for piping into seed filter

Expected yield from pass 2 (rough): ~50-80 new genuine recoveries from the
high-priority bucket, low double digits from medium, <10% from low. Total
expected new genuine-recovery rows: ~80-120, roughly doubling
`onpolicy_recovery_dataset.json`.

### Re-run command

```bash
# Filter seed_problems.jsonl to rerun candidates
python3 -c "
import json
ids = set(open('outputs/rerun_problem_ids.txt').read().split())
with open('seed_problems.jsonl') as f, open('outputs/rerun_seeds.jsonl', 'w') as out:
    for line in f:
        d = json.loads(line)
        if d['problem_id'] in ids: out.write(line)
"

# Re-run with the wider cap
python3 generate_injections.py \
    --seeds outputs/rerun_seeds.jsonl \
    --output_dir outputs/rerun \
    --max_rounds 2 \
    --max_tokens 16000 \
    --cumulative_token_cap 29000

# Filter the new traces with the same genuine-recovery criteria,
# then merge with onpolicy_recovery_dataset.json (dedupe by question + answer hash).
```
