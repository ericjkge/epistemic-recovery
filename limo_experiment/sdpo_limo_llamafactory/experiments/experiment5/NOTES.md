# Experiment 5 — On-policy injection-recovery training

Goal: build an on-policy SFT dataset by letting a model attempt competition-math
problems, then injecting a hedging phrase ("Wait, that might be wrong. Let's
try some different approaches.") at the end of any wrong attempt and
resampling — s1-style budget forcing applied to *correctness* rather than
length. Train a LoRA on the resulting traces and check whether it beats the
off-policy LIMO LoRA (experiments 1–4).

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
as in-trace markers — the model literally produced "Wait, that might be wrong.
Let's try some different approaches." each retry).

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
- **Injection phrase wording.** Current: `" Wait, that might be wrong. Let's
  try some different approaches.\n\n"`. s1 just used `"Wait"`. Earlier
  iterations used `"Wait, the answer is wrong. Let's think again."` (asserts
  wrongness as fact, drove a "check-and-reverify" pathology), and then
  `"Wait, that might be off. Alternatively, let me try a different
  approach."` (good but pluralized exploration not strong enough). The
  current wording softens with `might be wrong` (uncertainty rather than
  claim of error) and explicitly invites *plural* "different approaches" to
  push the model away from re-verifying its single prior path and toward
  open hedge-and-explore.
- **Cap on cumulative tokens.** We cap at 30,720 cumulative generated tokens
  (R1=20,000 + R2=10,720) in `generate_injections.py`, matching
  `cutoff_len=32768` in `qwen3_sdpo_lora_sft.yaml` (Qwen3-8B's native
  context). The asymmetric per-round split reflects that R1 has only the
  question as prompt (full context available for thinking) while R2's
  prompt is R1's full trace + injection (so its remaining vLLM context is
  ~32K - 20K - 250 prompt overhead - 16 injection ≈ 12.5K, of which we use
  10,720 with margin). `cumulative_token_cap` counts only generated tokens
  (excludes prompt and injection-phrase tokens). Round 2 worst-case vLLM
  context use is ~31K, inside `max_model_len=32768`.
- **Risk of "self-fulfilling" injection.** If the model learns that
  "Wait, that might be wrong. Let's try some different approaches." always
  precedes a correct answer in training data, it may emit the phrase
  liberally at inference and skip honest verification. Worth checking
  final-token distributions post-training.

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

Move to **`max_rounds=2`, `max_tokens_r1=20000`, `max_tokens_r2=10720`,
`CUMULATIVE_TOKEN_CAP=30720`** (R1+R2 sum to the cap) to:

1. Cut R1 truncation to near-zero (smoke test on 20 problems showed 0/20
   R1s without a `\boxed{}` at 18K; bumping to 20K covers the residual long
   tail), eliminating the budget-vs-injection confound on the recovery
   measurement.
2. Stay safely inside Qwen3-8B's 32K native context. R2 worst case is
   ~250 (chat template + question) + 20,000 (R1 thinking) + 16 (injection)
   + 10,720 (R2) ≈ 31K, with ~1.7K margin under `max_model_len=32768`.
3. Drop `max_rounds=3`. R3 contributed only 1.7pp pass@1 lift (53/3180
   chains) under the old config and most of that lift was budget-driven
   (R1 truncated → R2/R3 just finishes writing). With the wider R1 cap and
   genuine R1 commitment, R3 gives diminishing returns.

The asymmetric R1=20K / R2=10.7K split reflects that R1 has the full vLLM
context for thinking from scratch, while R2's prompt already contains
R1's trace and so has less room left. R2 is also a *targeted* recovery
(continue from a specific failure) rather than a fresh run, so it
generally needs fewer tokens.

**Existing pass-1 traces are kept** — they're already on disk in
`onpolicy_injection_dataset.json` and `onpolicy_recovery_dataset.json`,
and were used to identify the budget contamination. The pass-2 re-run
only adds new data; nothing lost.

## Results — generation pass 2

Run config: `model=baseline`, `max_rounds=2`, `max_tokens_r1=20000`,
`max_tokens_r2=10720`, `cumulative_token_cap=30720`, hedging phrase
`" Wait, that might be wrong. Let's try some different approaches.\n\n"`.

Seed set: `outputs/diagnostic_seeds.jsonl` — the 628 LIMO-v2 problems left
after dropping the 167 in `outputs/easy_r1_solved_ids.txt` (= problems
where all 4 chains got R1 correct in the overnight pass-1 run, so
injection can't help). Trace files:

- `outputs/pass1/` — `num_chains=1`, `seed=42`, 78 min wall
- `outputs/pass3/` — `num_chains=3`, `seed=43`, 247 min wall
- `outputs/pass4/` — combined via `combine_passes.py` (4 chains/problem)

### Pass rates

| Layer | R1 | R2 |
|---|---|---|
| Per-chain pass@1 (combined, 2512 chains) | 35.5% | **39.2%** (+3.7pp) |
| Problem-level pass@4 (any chain) | 64.3% | **66.9%** (+2.6pp) |

R1 truncation effectively eliminated: 0 chains hit `max_tokens_r1=20000`
in either pass; only **12/2512 chains** failed to emit a `\boxed{}` in R1
(0.5%, vs ~16% under the old 10K cap). R2 cumulative-cap hits: 14
(pass@1) + 56 (pass@3) = 70/2512 = 2.8% — these are chains that fully
spent both rounds and still didn't recover.

### Recovery dataset (R2-correct cohort)

| | n |
|---|---:|
| R2-correct chains (any) | 90 |
| ↳ pure truncation (no R1 \\boxed) | 12 |
| ↳ at-cap with answer | 1 |
| **↳ genuine committed-wrong → recovered** | **77** |
| Distinct problems with ≥1 genuine recovery | 67 |

Genuine-recovery rate of 85.6% of R2-correct (vs ~72% under the old
18K-cap config), confirming the wider R1 budget produces purer recovery
signal at the cost of overall recovery count. R1-correct rows available
for SFT diversity: 894 across the 2512 chains.

### Epistemic density (pooled, per 1K thinking words)

Headline only — see `analyze_epistemic_tokens.py` for the full per-token
table. On the R2-correct cohort (n=90): R1=16.49, R2-marginal=12.66,
FULL=15.61. R2 content is ~23% less hedge-dense than R1, which we read
as targeted recovery rather than open re-exploration. The injection's
phrase tokens (`wait`, `might`) drive a +58% lift in `might` density in
R2, but `alternatively` / `maybe` / `perhaps` all *fall* in R2 — the
"different approaches" pluralization didn't broaden exploration tokens
the way the wording suggested it might.

### SFT data outlook

77 genuine recovery rows is below the ~150-problem floor for stable LoRA
SFT. Options: (a) augment with a stratified sample of R1-correct rows
for diversity, (b) run another seeded pass to grow the genuine bucket,
or (c) iterate on the phrase to boost `alternatively` / `maybe` in R2
before bulking up.

## Running detached for long jobs

A full 795-problem pass@1 takes ~80 min; pass@3 takes ~3 hours. Launching
through an IDE-tied shell (Cursor / Claude Code) means the run can die
when the IDE restarts. Use `setsid` + `nohup` + redirects to fully detach
the process so it's owned by `init` (PID 1) and survives any parent exit:

```bash
cd /home/ubuntu/epistemic-recovery/limo_experiment/sdpo_limo_llamafactory/experiments/experiment5
setsid nohup python3 generate_injections.py \
    --seeds outputs/diagnostic_seeds.jsonl \
    --output_dir outputs/pass1 \
    --num_chains 1 \
    --model baseline \
    --seed 42 \
    < /dev/null > outputs/pass1.log 2>&1 &
disown
```

Each piece matters:
- `setsid` → new session, decouples from controlling terminal
- `nohup` → ignore SIGHUP if the parent shell does end up signalling its
  process group
- `< /dev/null` → no stdin (otherwise an IDE-closed terminal can send EOF
  and stall reads)
- `> outputs/<run>.log 2>&1` → both streams to a file we can tail
- `&` → background
- `disown` → remove from the shell's job table so the shell can exit
  without the kernel sending SIGHUP to children

### Verify the run is alive

```bash
pgrep -af "python3 generate_injections"     # should show the python parent
ps -ef | grep VLLM | grep -v grep           # should show the EngineCore child
tail -f outputs/pass1.log                    # live log
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```

### Kill cleanly

`pkill -f "generate_injections"` is dangerous from inside an interactive
shell because the bash command line *itself* contains that substring,
so pkill matches and kills your own shell mid-command. Anchor the regex
at the start of the cmdline so it only matches the python process:

```bash
pkill -f "^python3 generate_injections"
```

Or kill by PID directly: `pgrep -f "^python3 generate_injections" | xargs -r kill`.
After killing, also check for orphaned `VLLM::EngineCore` processes
(child of the parent python; if SIGTERM-ed parent dies first the engine
can be reparented to init and keep using the GPU): `pgrep -af "VLLM::EngineCore"`
and kill any matches by PID.
