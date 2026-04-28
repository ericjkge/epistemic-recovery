# Epistemic Case Studies: SDPO + LIMO LoRA vs Baseline SDPO

Two AIME 2025 problems where the LoRA-tuned model (SDPO base + LIMO LoRA SFT) reliably solved a problem the baseline SDPO model could not. In both cases, the LoRA's win is traceable to a specific verbalization-and-recheck pattern that LIMO fine-tuning appears to have transferred.

- Models: `beanie00/math-SDPO-Qwen3-8B-think-step-100` (baseline) vs same + LoRA from [qwen3_sdpo_lora_sft.yaml](../qwen3_sdpo_lora_sft.yaml) (rank 16, 2 epochs, 760 LIMO traces).
- Eval: AIME 2025, n_sampling=4 per problem.
- Source data: [lora_sdpo_plus_limo_aime25.json](lora_sdpo_plus_limo_aime25.json), [baseline_sdpo_think_step100_aime25.json](baseline_sdpo_think_step100_aime25.json).

---

## Case 1 — AIME 2025 Problem 23 (off-by-one at an excluded boundary)

### Question

> There are $n$ values of $x$ in the interval $0 < x < 2\pi$ where $f(x) = \sin(7\pi \sin(5x)) = 0$. For $t$ of these $n$ values of $x$, the graph of $y = f(x)$ is tangent to the $x$-axis. Find $n + t$.

Ground truth: **149**.

### Statistics

| Model | Sample 0 | Sample 1 | Sample 2 | Sample 3 | acc | any-correct |
|---|---|---|---|---|---|---|
| Baseline SDPO | 150 ❌ | 36 ❌ | 150 ❌ | 150 ❌ | 0.00 | False |
| SDPO + LIMO LoRA | **149 ✓** | 150 ❌ | **149 ✓** | 150 ❌ | 0.50 | True |

The baseline locks onto **150** in 3/4 samples. The LoRA produces **149** in 2/4. The error is exactly off-by-one: the "obvious" answer is 150, the correct answer is 149.

Epistemic-marker counts (number of "wait", "hmm", "hold on", "let me check", "but actually", "boundary", "edge", "endpoint", "careful" tokens in each trace):

| Trace | Markers | Final answer |
|---|---:|---|
| Baseline sample 0 | 4 | 150 |
| LoRA wrong sample (1) | 18 | 150 |
| LoRA correct sample (0) | **47** | **149** |

The correct LoRA sample questions ~12× more than the baseline. The wrong LoRA sample is in between: the LIMO style is on, but it doesn't trigger the right re-derivation.

### Where the off-by-one comes from

Setting $\sin(7\pi \sin 5x) = 0$ requires $\sin 5x = k/7$ for integer $k$ with $|k| \leq 7$. For each non-zero $k$ with $|k| < 7$, the equation $\sin 5x = k/7$ has 2 solutions per period of $\sin 5x$ (period $2\pi/5$, so 5 periods over $0 < x < 2\pi$) → 10 solutions. For $k = \pm 7$ ($\sin 5x = \pm 1$), each gives 5 solutions. **For $k = 0$, the naive count gives 10, but the correct count is 9** because the zero at $x = 2\pi$ falls on the excluded endpoint of the open interval $0 < x < 2\pi$.

- Naive: $12 \cdot 10 + 2 \cdot 5 + 10 = 140$, $t = 10$, sum = **150**.
- Correct: $12 \cdot 10 + 2 \cdot 5 + 9 = 139$, $t = 10$, sum = **149**.

### Critical epistemic moment (LoRA correct sample, ~chars 5700–6500)

The LoRA derives "5 periods × 2 = 10 solutions" from the period count, then *immediately cross-checks by manual enumeration* and finds 9. It then localizes the discrepancy to the excluded endpoint:

> "...so 5×2 = 10 solutions? **Wait, but when I count manually**, for $x = \pi/5, 2\pi/5, ..., 9\pi/5$, that's 9 solutions. **Hmm, so maybe my previous reasoning was wrong here.**
>
> **Wait, let me think again.** If the period is $2\pi/5$, then over the interval $0 < x < 2\pi$, there are 5 periods. In each period, the sine function crosses zero twice... So over the entire interval $0 < x < 2\pi$, the zeros would be at:
>
> - First period: π/5, 2π/5
> - Second period: 3π/5, 4π/5
> - Third period: 5π/5 (π), 6π/5
> - Fourth period: 7π/5, 8π/5
> - **Fifth period: 9π/5, 10π/5 (but 10π/5 = 2π is excluded)**
>
> Wait, so actually, in each of the first four periods, there are two zeros, but **the fifth period only has one zero at 9π/5**. So total zeros would be 2+2+2+2+1 = 9. So that's why there are 9 solutions for $k = 0$."

The same trace contains 47 such "wait / let me check / hmm" markers; the LoRA wrong sample contains 18, and the baseline wrong sample contains 4.

### What the baseline does instead

Baseline sample 0 makes the period-count argument and accepts it without enumerating. The only late-trace re-check is shallow:

> "...and the number of tangent points is $t = 10$, so $n + t = 140 + 10 = 150$. **Wait, but earlier I thought that for $k = \pm 7$, each gives 5 solutions, totaling 10**, and for other $k$, each gives... [restates the same calculation, doesn't recount k=0] ...total $n = 140$. And $t = 10$. So the answer is 150. That seems correct."

The "wait" exists, but it's a confirmation, not a re-derivation. Endpoint exclusion at $x = 2\pi$ is never examined.

### What LIMO appears to have transferred

1. Validating a closed-form count against a manual enumeration even when the closed-form looks correct.
2. Treating a small numerical disagreement (9 vs 10) as a load-bearing signal rather than a bookkeeping detail.
3. Re-examining whether boundary points are inside or outside the open interval.

---

## Case 2 — AIME 2025 Problem 17 (multiplicity in casework, not just enumeration)

### Question

> Four unit squares form a $2\times 2$ grid. Each of the 12 unit line segments forming the sides of the squares is colored either red or blue in such a way that each unit square has 2 red sides and 2 blue sides. Find the number of such colorings.

Ground truth: **82**.

### Statistics

| Model | Sample 0 | Sample 1 | Sample 2 | Sample 3 | acc | any-correct |
|---|---|---|---|---|---|---|
| Baseline SDPO | 16 ❌ | 8 ❌ | 12 ❌ | 6 ❌ | 0.00 | False |
| SDPO + LIMO LoRA | **82 ✓** | **82 ✓** | (no extracted) ❌ | **82 ✓** | 0.75 | True |

The baseline's wrong answers (16, 8, 12, 6) are all small numbers consistent with **undercounting by ignoring multiplicities** — i.e., treating each casework branch as contributing one coloring instead of summing 1's and 2's. The LoRA gets 82 in 3 of 4 samples.

Epistemic-marker counts:

| Trace | Markers | Final |
|---|---:|---|
| Baseline sample 0 | 184 | 16 |
| Baseline sample 1 | 146 | 8 |
| LoRA correct sample 0 | **388** | 82 |
| LoRA correct sample 1 | **650** | 82 |

The baseline is **not low on self-questioning here** (184 markers — far more than its problem-23 trace). It questions plenty. The failure mode is different: it questions, almost reaches the right structure, then collapses to a wrong shortcut.

### The structural insight needed

Label edges $x_1, \ldots, x_{12}$. The four "2-red-per-square" constraints reduce to four equations of the form $x_i + x_j = 2 - x_a - x_b$, where $(x_a, x_b)$ are shared (interior) edges. After fixing the 4 interior edges (16 configurations), each remaining pair $(x_i, x_j)$ must sum to a value $k \in \{0, 1, 2\}$, and the **number of binary solutions for that pair depends on $k$**:

- $k = 0$: 1 way (both 0)
- $k = 1$: **2 ways** (one of each)
- $k = 2$: 1 way (both 1)

Across 4 such pairs and 16 interior configurations, summing the products gives **82**.

### Critical epistemic moment (LoRA correct sample, ~chars 19400–20700)

The LoRA explicitly states the multiplier table for the pair sum and applies it:

> "Let me define variables $a = x_1 + x_7 = 2 - x_8 - x_3$,
> $b = x_2 + x_9 = 2 - x_8 - x_4$,
> $c = x_5 + x_{10} = 2 - x_3 - x_{11}$,
> $d = x_6 + x_{12} = 2 - x_4 - x_{11}$ ...
>
> Wait, but actually, **for two variables, the number of solutions for their sum being $k$ is**:
> - **If $k=0$: both are 0 → 1 solution**
> - **If $k=1$: one is 1, the other 0 → 2 solutions**
> - **If $k=2$: both are 1 → 1 solution**
>
> Therefore, for each pair, the number of solutions for their sum being $k$ is 1, 2, 1 for $k=0,1,2$ respectively."

This is the load-bearing observation. Once it's stated cleanly, the rest is mechanical: enumerate the 16 interior configurations and sum the per-configuration products of pair-multipliers.

### What the baseline does instead

The baseline's sample 0 actually *encounters* the same multiplicity issue — multiple times — and oscillates without resolving:

> [chars ~28311] "Recall that in the initial analysis, I considered the number of valid colorings as 16 (shared segments) * 2 (boundary segments) = 32. But in reality, **for each shared segment configuration, the number of boundary segment configurations is not always 2. It depends on the shared segments**."
>
> [chars ~28799] "However, in the example where H1=0, H2=0, V1=1, V2=1, there were 2 solutions, and for others, 1 or 2..."
>
> [chars ~30167] "For each of the 4 shared segments, there are 2 choices, so 16 configurations. For each, the number of valid boundary segment configurations is 2 if the equations are satisfied, and 1 otherwise."
>
> [chars ~42069, attempting casework] "Case 3: 2, Case 4: 2, Case 5: 1, Case 7: 2, Case 8: 2, Case 9: 2, Case 10: 2, Case 12: 1, Case 13: 2, Case 14: 2 → That's 2+2+1+2+2+2+2+1+2+2 = 16."
>
> [final, chars ~50500] "**Summing over all 16 shared segment configurations, we find that each configuration leads to exactly one valid coloring. This is because the constraints are over-constrained, and for each shared segment configuration, the boundary segments are uniquely determined.** Thus, the total number of valid colorings... is $\boxed{16}$."

Three failures are visible:

1. **Arithmetic collapse.** The casework partial sum "2+2+1+2+2+2+2+1+2+2" actually equals 18, not 16, but the baseline writes "= 16" and moves on. It questions a lot but doesn't audit its own arithmetic.
2. **Premature commitment.** After noting "1 or 2 depending on configuration," the baseline reverts to "exactly one valid coloring per configuration" with the false justification "constraints are over-constrained."
3. **Casework abandonment.** Started enumerating 16 cases, partially completed, then declared an unverified shortcut.

### What LIMO appears to have transferred

1. Stating the multiplier table $\{1, 2, 1\}$ explicitly before applying it, rather than implicitly assuming multiplicity 1.
2. Following through on a casework enumeration once started, rather than collapsing to a heuristic when the bookkeeping gets long.
3. Resisting the "constraints over-determine the answer" shortcut when partial evidence already shows variable multiplicity.

---

## Cross-case pattern

Both wins look like the same intervention from different angles:

- **Problem 23**: don't trust a closed-form count without checking it against an enumeration; respect open-interval boundaries.
- **Problem 17**: don't trust a "structurally unique" shortcut when the casework already produced different multiplicities; complete the enumeration.

In both cases, the baseline does *some* self-questioning. The LoRA does substantially more (3–12× the marker count), and — more importantly — its self-questioning ends with **a recomputation from a different angle that reaches a different number**, instead of a restatement that confirms the original number. This "verbalize, then re-derive, and let the second derivation override the first if they disagree" pattern is the most concrete behavior that LIMO fine-tuning seems to have transferred.

### Caveats

- 4 samples per problem is a small base. The 0/4 vs 3/4 contrast on problem 17 and 0/4 vs 2/4 on problem 23 is suggestive but not statistically tight.
- Marker counts measure verbalization volume, not verbalization quality. The LoRA wrong sample on problem 23 has 18 markers and still gets 150 — verbalization is necessary but not sufficient.
- Problem 17 baseline arrives at "82-adjacent" reasoning before collapsing. The LoRA's improvement may be partly about *committing* to enumeration as much as about reasoning style.
