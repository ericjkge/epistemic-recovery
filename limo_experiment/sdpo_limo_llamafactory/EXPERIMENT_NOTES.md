# Experiment notes — SDPO + LIMO LoRA recovery (run 2026-04-24)

Working notes from the first end-to-end run. Captures what the data actually shows and what
needs to change next time. All numbers from `results/*.json` at n=4 per problem.

---

## Headline findings

1. **LoRA recovers epistemic verbalization — strongly.** Avg epistemic tokens per response
   (sum of the Kim et al. 10-token set inside `<think>`):
   - baseline SDPO: 55.5
   - + LoRA on LIMO: 290.5  (**+424%**)

2. **Pass@N improves more than per-sample accuracy.** This is the signature you'd predict
   if epistemic verbalization restores *exploratory* reasoning:

   | | baseline | LoRA | Δ |
   |---|---|---|---|
   | acc@4 (≈ pass@1) mean | 0.521 | 0.567 | +4.6pp |
   | any-correct (pass@4) mean | 0.650 | 0.800 | **+15pp** |

3. **Coverage gain is concentrated out-of-distribution.** LIMO is pre-2024 competition math;
   AIME25 is held-out.

   | | only-LoRA solved | only-baseline solved |
   |---|---|---|
   | aime24 | 3 | 0 |
   | aime25 | **7** | 1 |

   Net pass@4 gain: +9 problems across both benchmarks. AIME25 gained 2× what AIME24 did.

4. **Hidden cost: ~25% of LoRA responses are degenerate repetition loops.** Counting trigram
   repetition >30% in the last 8K tokens:

   | | loops | loops % | accuracy within loops |
   |---|---|---|---|
   | baseline combined | 14/240 | 5.8% | 0/14 |
   | LoRA combined | **61/240** | **25.4%** | 4/61 |

   LoRA has ~4× baseline's loop rate. Loops are near-zero accuracy. Invisible in headline
   metrics — shows up only when you look at tail behavior or per-sample text.

---

## Methodological issues surfaced

1. **`<think>` regex silently falls back to full response** when `</think>` is missing. Inflates
   epistemic counts for truncated samples. Fixed in [eval/analyze_epistemic.py](eval/analyze_epistemic.py):
   `extract_thinking_span` now treats unclosed `<think>` as "thinking span = everything after
   the opening tag" — apples-to-apples with closed spans.

2. **Initial `max_tokens=38912` was misdiagnosed as the bottleneck.** Raising it to 65K would
   have just given loops more room to run. Only ~3% of truncations were genuine long reasoning
   (`aime25_020`-style mid-polynomial cutoffs); ~80–86% were stuck repeating the same trigram.
   The fix is `frequency_penalty`, not a larger cap. Current defaults in
   [eval/evaluate_aime.py](eval/evaluate_aime.py): `max_tokens=24576`, `frequency_penalty=0.3`.

3. **`n_sampling=4` is too noisy for confident claims.** Each any-correct rate has ±1/30 ≈
   ±3.3pp quantization per problem, and pass@4 has intrinsic sampling variance on top. The
   "LoRA acc@4 slightly worse on aime24" finding (0.600 vs 0.617) is well within noise.

4. **No `pretrained` Qwen3-8B run.** Without the upper-bound, "LoRA recovers X% of the
   suppression gap" can't be computed — only "LoRA is +424% vs baseline." The hypothesis
   under test is that LoRA recovers toward the pretrained distribution, not just "makes
   counts go up."

5. **vLLM LoRA/Punica segfaults on H100 + CUDA 12.4 + driver 580.** Worked around by merging
   the adapter pre-eval ([eval/merge_lora.py](eval/merge_lora.py)). If next server hits this,
   the merged-model path is reliable. If not, test plain LoRA load first for speed.

---

## Recommendations for future runs

### Training-side

- **Consider LIMO + broader corpus.** 813 examples is tight; loops may be the model
  memorizing LIMO's long-verification style without learning to terminate. A 1–2K example mix
  (LIMO + DAPO subset) could cure the loop pathology at the source.
- **Try 1 epoch instead of 2.** Current config runs 2 epochs — at 813 examples that's already
  low-data SFT territory. 1 epoch might preserve recovery while reducing overfitting to the
  long-trace style.
- **Filter LIMO tails.** LIMO response length has heavy right tail (max ~76K chars ≈ 20K
  tokens). Training on responses >15K tokens may be what teaches "keep verifying forever."
  An easy ablation: cap training examples at, e.g., 10K tokens.

### Inference-side (already applied, but document the reasoning)

- **`frequency_penalty=0.3` by default.** Small enough to not harm legitimate "wait, let me
  check" behavior; large enough to make literal trigram loops unstable. Worth ablating
  `{0.0, 0.15, 0.3, 0.5}` once.
- **Keep `max_tokens` in the 20–25K range.** Baseline p95 is ~23K; anything larger is
  accommodating pathology, not real reasoning.
- **Bump `n_sampling` to 16.** Brown et al.'s unbiased pass@k estimator wants at least 10–16
  samples for tight bounds. Compute cost: 4× current.

### Evaluation-side

- **Track loop rate as a first-class metric.** Already clearly visible in the data (25% vs
  6%). Should go in the summary CSV and the qualitative summary alongside accuracy.
- **Report an "effective" pass rate.** Discount looped responses the same way you'd discount
  responses with no answer. Current numbers give LoRA credit for responses that repeat
  `3+4+7=14?` 500 times.
- **Add the `pretrained` run.** Needed to make the "recovery %" claim.
- **Stratify epistemic counts by closed vs unclosed spans.** Related to the regex fix — even
  after the fix, truncated LoRA samples contribute much longer spans and therefore higher
  epistemic counts. Reporting per-1K-token rates would be more honest.

### Analysis-side

- **Always do per-benchmark in vs OOD split.** The in/OOD story (aime24 ≈ training-era vs
  aime25 = held-out) was only visible in the benchmark breakdown. A mean-across-benchmarks
  number would have hidden the strongest evidence for the hypothesis.
- **Save token-level logprobs during generation.** The token-distribution analysis
  ([eval/analyze_token_distribution.py](eval/analyze_token_distribution.py)) currently needs
  a separate teacher-forced pass. Asking vLLM for `logprobs=1` at generation time would let
  us compute per-token entropy at epistemic positions directly on the eval outputs — a much
  cleaner signal than counts.
- **Keep `analyze_epistemic.py` in the eval venv.** It only needs stdlib + matplotlib. No
  need to set up `.venv-train` just for analysis, despite what the original README said.

---

## Open questions worth answering next

1. Does `frequency_penalty` alone eliminate the loop mode, or does it just push it into a
   different pathology (e.g., truncated answers without `\boxed{}`)?
2. Is the LoRA accuracy gain *dependent* on the loop samples, or does it hold on the
   clean-response subset? (Stratify pass@4 by loop/clean.)
3. If the model solves a problem only at pass@N > 1, what fraction of those responses contain
   epistemic-token spikes in the successful sample vs the failed ones? (Within-problem
   correlation between epistemic verbalization and correctness.)
4. Does the loop pathology show up in the token-distribution analysis as a collapse in
   entropy? If so, `analyze_token_distribution.py` could be used as a pre-inference filter.
5. On baseline, 14/240 are also loops — but they're not hurting accuracy at 0%. Are those
   baseline loops on *different* problems than LoRA's loops, or the same ones? If the same,
   LoRA might be inducing loops mostly on the already-hard questions (which would partly
   explain why "only-LoRA-solved" list is dominated by L 1/4 scores).
