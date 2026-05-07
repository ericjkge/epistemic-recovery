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
