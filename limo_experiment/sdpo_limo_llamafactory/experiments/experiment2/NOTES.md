# Experiment 2 — SDPO + LIMO LoRA, n=8 AIME25 evaluation

Run ID `20260427_195403`. Adapter at `adapter/`, merged model at `adapter/merged/`. Eval data
at `eval_results/`.

---

## Headline numbers (AIME25 only, n=8 per problem, 30 problems)

| metric | baseline SDPO | + LoRA | Δ |
|---|---:|---:|---:|
| pass@1 (≈ acc per sample) | 0.392 | **0.500** | +10.8pp |
| **pass@4** (Chen et al. unbiased) | 0.606 ± 0.079 | **0.734 ± 0.072** | **+12.8pp** |
| pass@8 (any-correct) | 0.700 ± 0.084 | **0.800 ± 0.073** | +10.0pp |
| avg response length (chars) | 9 154 | 13 721 | +50% |
| avg epistemic tokens / response | 61.3 | **219.4** | **+258%** |

`pass@k` computed with the unbiased estimator from Chen et al. 2021:
`pass@k = 1 − C(n−c, k) / C(n, k)`. SEs are SE-of-mean across the 30 problems.

The +12.8pp pass@4 gap clears its ±0.08 SE comfortably, so the LoRA-helps-coverage finding
is real on this run, not noise — addressing the "well within noise" concern from
EXPERIMENT_NOTES re: experiment 1's n=4 measurements.

`pass@k` saturates faster on LoRA (0.50→0.73→0.80, decelerating) than baseline
(0.39→0.61→0.70). LoRA's coverage advantage is concentrated at small k; past k=4 the
baseline catches up. Consistent with the "stochasticity boost" reading — LoRA explores
more diverse trajectories per attempt rather than producing higher-quality individual
samples.

---

## Coverage delta — what LoRA actually unlocks

| | count | problem ids |
|---|---:|---|
| LoRA-only solved | **3** | aime25_006, aime25_010, aime25_023 |
| Baseline-only solved | **0** | — |

The flips are one-directional: every problem the baseline can solve, the LoRA can also
solve. The +3 net is pure coverage gain.

---

## Epistemic verbalization — methodological wrinkle

`avg_epistemic_per_response` initially returned **0** for the LoRA model. Investigation
revealed the LoRA emits an **empty `<think></think>` block in 100% of samples** and puts
all reasoning *outside* the tags. Our training data
([make_limo_dataset.py](../../make_limo_dataset.py)) writes the raw LIMO solution as the
output target with no `<think>` wrapping — the LoRA learned that style and stopped using
thinking mode.

`extract_thinking_span` in [analyze_epistemic.py](../../eval/analyze_epistemic.py) was
patched to detect the empty-think case and use the post-`</think>` reasoning as the
thinking span (trimming the trailing `\boxed{...}` answer line for length comparability).
After the fix, LoRA's epistemic count came back at the expected ~3-4× baseline.

**Implication for future training**: re-introduce `<think>...</think>` wrapping on the
training data so the adapter preserves Qwen3 thinking-mode structure. The deleted
`prepare_limo.py` had the right wrapping logic but used the older GAIR/LIMO + ShareGPT
format; would need a small rewrite to apply that logic on top of the current alpaca
pipeline used by `make_limo_dataset.py`.

### Per-token breakdown (mean count per response)

| token | baseline | LoRA | LoRA/baseline |
|---|---:|---:|---:|
| wait | 22.6 | **56.3** | 2.5× |
| alternatively | 5.9 | **47.8** | 8.1× |
| perhaps | 3.2 | **43.1** | 13.5× |
| maybe | 7.9 | **26.4** | 3.3× |
| hmm | 0.6 | **13.8** | 21.5× |
| actually | 1.7 | **13.4** | 7.9× |
| check | 6.4 | 8.9 | 1.4× |
| might | 6.9 | 5.3 | 0.8× ↓ |
| seems | 3.8 | 3.5 | 0.9× ↓ |
| likely | 2.2 | 0.7 | 0.3× ↓ |

Strong effect on hypothesis-revision tokens (`alternatively`, `perhaps`, `actually`,
`hmm`). Mild regression on hedging tokens (`might`, `seems`, `likely`). Distribution shift
is consistent with re-acquiring "exploratory reasoning" style rather than uniformly more
hedging.

---

## Degenerate repetition loops

Loop rate is sensitive to the window size used for trigram-repetition; reporting both:

| window | baseline | LoRA | correct-in-loops (baseline / LoRA) |
|---|---:|---:|---:|
| last 8 000 chars (~2K tokens, EXPERIMENT_NOTES default) | 17.5% | 21.2% | 29% / 14% |
| last 8 000 words (~8K tokens) | 51.2% | 49.2% | 19% / 36% |

Two readings:

- **Tight tail (chars)**: LoRA has marginally more pathological short-window loops
  (21% vs 17%) and correctness within those loops is much lower (14% vs 29%). The 25%
  loop-rate finding from experiment 1's EXPERIMENT_NOTES replicates *directionally*
  — LoRA still loops more in the worst-case tail.
- **Wide tail (words)**: When measured over ~8K tokens, LoRA has *slightly fewer* loops
  but those loops are far more often correct (36% vs 19%). This suggests many of LoRA's
  repetitions are verification-style ("let me check ... actually wait ...") that
  successfully arrives at an answer, not the dead-end repetitions baseline produces.

Either way, repetition pathology is not gone. It's a softer version of experiment 1's
"~25% loops, ~0% correct in loops" finding — but still the dominant remaining pathology.

---

## Where LoRA wins (qualitative)

First correct generation on each LoRA-only-solved problem (epistemic count in the
generation):

- **aime25_006** (190 epistemic tokens): probabilistic reasoning. LoRA reframes the
  problem multiple times — *"Alternatively, maybe I can think of the problem as follows:
  the probability that the last word contains G is equal to the probability that, in the
  pairing, the pair containing G is the highest pair."* Baseline got stuck on a single
  framing.
- **aime25_010** (202 epistemic tokens): identifies extraneous solutions —
  *"But the problem says 'finitely many points', which is true, but maybe some of these
  solutions are extraneous?"* This is the kind of "wait, hold on" check that suppression
  removes.
- **aime25_023** (92 epistemic tokens): explicit miscalculation check —
  *"Hmm, but let me check if there's a miscalculation here."*

Mean LoRA-only-solved generation length: **44 861 chars** — substantially longer than the
overall average. LoRA is willing to spend tokens on hypothesis revision.

---

## Limitations of this run

- **Single benchmark**: only AIME25 was re-evaluated at n=8. Experiment 1's broader
  pattern (gain stronger on AIME25 vs AIME24, "concentrated OOD") is consistent with
  these numbers but unconfirmed at n=8.
- **No upper bound**: pretrained Qwen3-8B not run here, so "LoRA recovers X% of the
  suppression gap" is still uncomputable on this run.
- **Same caveat as experiment 1**: pre/post-think-tag measurement asymmetry was
  invisible until inspected directly. Future runs should make `<think>`-tag presence
  a first-class diagnostic.

---

## Next steps

1. **Add `<think>` wrapping to LIMO training data.** Highest-leverage fix. Restoring
   thinking-mode structure preserves downstream tool compatibility (probe scripts,
   Kim et al.'s analyses) while keeping the epistemic-content gain. Adapt the wrapping
   logic from the deleted `prepare_limo.py` onto `make_limo_dataset.py`'s alpaca
   pipeline.
2. **AIME24 at n=8.** Replicate the n=8 measurement on AIME24 to confirm the
   "OOD-concentrated" finding. ~30 min if you skip the baseline (already on disk from
   experiment 1, though only at n=4 — would need a re-run for apples-to-apples).
3. **Pretrained Qwen3-8B upper bound.** Run AIME25 + AIME24 at n=8 on the unsuppressed
   base. Lets us frame results as "LoRA recovers X% of the gap" rather than "LoRA is
   +Y% over baseline."
4. **Hparam sweep (the actual experimentation phase, see prior planning notes).** With
   the empty-`<think>` issue understood and pass@4 properly computed, the sweep can
   target loop rate as a first-class metric with cleaner signal.
5. **Bake `pass@4` and loop rate into `analyze_epistemic.py`.** Both metrics should be
   in the standard CSV output going forward, not computed ad-hoc.
6. **Investigate the "lora correct-in-loops at wide window" finding.** That 36%-correct
   rate inside wide-window "loops" suggests verification-style repetition is *productive*
   for LoRA but not for baseline. Worth a closer look — could imply different
   anti-repetition strategies (e.g. lower `frequency_penalty` for LoRA than baseline).
