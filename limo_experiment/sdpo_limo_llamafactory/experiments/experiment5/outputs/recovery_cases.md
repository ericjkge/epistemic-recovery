# Genuine Epistemic Recovery — Example Traces

Examples of "clean" recoveries from `injection_traces.jsonl`: chains where the
baseline SDPO model committed to a definite (boxed) wrong answer, received the
injection prompt `"Wait, the answer is wrong. Let's think again."`, and then
arrived at the correct answer in a subsequent round.

These are the kinds of traces collected in `onpolicy_recovery_dataset.json`
(126 rows total — 80 Case A + 46 Case B).

---

## Case A — wrong R1 → correct R2 (80 traces)

The model commits a wrong answer in round 1, the injection forces a re-attempt,
and round 2 produces the correct answer.

### A.1 — `limo_seed_0027` ch 0 (4×4 seating, GT = 11)

**Question.** In a 4×4 meeting-room grid, no three people may sit consecutively
in a row or column. Find the maximum number of people seated.

**R1 commits 8 (wrong).** The model proposes a clean alternating pattern with
two people per row / per column:

> Row 1: X, _, X, _
> Row 2: _, X, _, X
> Row 3: X, _, X, _
> Row 4: _, X, _, X
>
> This gives 8 people, with no three in a row or column. ... Hence, the
> maximum number of participants is 8.
> **Final Answer** \boxed{8}

**Injection → R2 finds 11 (correct).** R2 immediately questions the implicit
"two per row/column" assumption:

> Wait, I thought earlier that each column can have at most two people, but
> maybe that's not the case. Let me think again. Suppose I have a column with
> three people. If they are not consecutive, like positions 1, 3, and 4,
> that's allowed.

The relaxed constraint (≤2 *consecutive*, not ≤2 total) lets R2 fit 11 people.

> Therefore, **11 people is the maximum** that satisfies the given conditions.
> $$\boxed{11}$$

**Recovery type.** Constraint misreading. R1 imposed a stricter rule than the
problem actually required; injection caused the model to re-read the constraint.

---

### A.2 — `limo_seed_0086` ch 1 (3×3×3 Rubik's cube, GT = 14)

**Question.** Maximum number of cells on a 3×3×3 cube such that no two marked
cells share a vertex.

**R1 commits 13 (wrong).** The model derives a clean parity-coloring argument
(cells with the same parity of `x+y+z` cannot share a vertex), counts the
classes, and picks the *smaller* one:

> So even sums: 12 + 1 = 13. Odd sums: 8 + 6 = 14. ... Therefore, the maximum
> number of cells we can mark without sharing a vertex is 13.
> **Final Answer** \boxed{13}

**Injection → R2 finds 14 (correct).** R2 spots the inconsistency in its own
work — the framework is right, but the wrong class was reported:

> Wait, I thought that the number of even sums is 13 and odd sums is 14. But
> earlier, I thought that if we color based on parity, then the cells with
> the same parity don't share a vertex. So if we take the larger group, which
> is 14, that would be the maximum.

R2 also sanity-checks the parity argument against (1,1,1) ↔ (2,2,2) and
re-derives the count cleanly.

> $$\boxed{14}$$

**Recovery type.** Last-step inversion. The framework was correct; the model
read off the wrong class. Injection prompted a recount.

---

### A.3 — `limo_seed_0140` ch 3 (power tower vs. nested factorials, GT = 104)

**Question.** Find the smallest `n` such that a tower of `n` 2s exceeds 100
nested factorials of 100!.

**R1 commits 102 (wrong).** The model identifies that `a_k ≈ 2↑↑(k+1)` and
concludes:

> Therefore, $a_{100} \approx 2 \uparrow\uparrow 101$. ... the minimal $n$
> is 102.
> **Final Answer** \boxed{102}

**Injection → R2 finds 104 (correct).** R2 spots a subtle bug in the
asymptotic identification:

> earlier, I thought that each factorial adds a layer, but actually, each
> factorial adds a layer, but the base of the tower is not 2, so the number
> of layers needed to surpass it is more.

R2 redoes the chain `a_1 < f(5)`, `a_2 < f(6)`, ..., `a_{100} < f(104)`.

> Therefore, the correct answer is 104.
> **Final Answer** \boxed{104}

**Recovery type.** Structural rethink. R1 used the wrong asymptotic
relation; injection caused the model to re-examine the base of the tower
and find a more careful bound.

---

## Case B — wrong R1 → wrong R2 → correct R3 (46 traces, 29 strict)

The model commits a wrong answer twice (in R1 and R2) before the second
injection finally lands on the correct answer in R3.

### B.1 — `limo_seed_0386` ch 3 (parallelogram area, GT = 8)

**Question.** Find the area of parallelogram ABCD enclosed in a figure (the
text-only version of the problem omits the figure, so the model must guess
plausible coordinates).

**R1 commits 12 (wrong).** Pure pattern-matching against common textbook
parallelograms:

> If the base is 4 and height 3, area is 12. If the base is 6 and height 2,
> area is 12. ... So 12 is a common answer. ... I'll go with that.
> **Final Answer** \boxed{12}

**Injection #1 → R2 commits 4 (still wrong).** R2 invents a specific coordinate
example and computes the shoelace area for it:

> Wait, the user might have had a specific problem in mind, like coordinates
> A(0,0), B(2,0), C(3,2), D(1,2). Using shoelace formula:
> ... Area = ½ |10 - 2| = ½ * 8 = 4. So area is 4.
> **Final Answer** \boxed{4}

**Injection #2 → R3 commits 8 (correct).** R3 abandons the invented-coordinates
strategy and tries different size combinations consistent with common variants
of the problem:

> I think the original problem might have been from a specific source where
> the area is 8. ... if the problem is about a parallelogram with base 4 and
> height 2, area is 8. ... I'll go with 8.
> **Final Answer** \boxed{8}

**Recovery type.** Search through a hypothesis space. With no figure
available, the model is essentially guessing plausible problem variants;
injections push it through three different hypotheses until one matches GT.
This is a borderline-genuine recovery — useful as SFT signal for "keep
revising your guess" but not the sharp self-correction shown in A.2 / A.3.

---

### B.2 — `limo_seed_0531` ch 2 (cube diagonal-bisecting plane, GT = 19)

**Question.** A 3×3×3 cube is bisected by a plane perpendicular to one of its
internal diagonals. How many unit cubes does the plane intersect?

**R1 commits 22 (wrong).** The model counts unit cubes whose corner-coordinate
sums straddle 4.5 (the bisecting plane in normalized coordinates):

> The plane cuts through the cube, and the number of intersected cubes can be
> calculated by considering how many cubes have at least one corner on each
> side of the plane. ... the detailed count gives 22.
> **Final Answer** \boxed{22}

**Injection #1 → R2 still commits 22 (wrong).** R2 attempts a layer-by-layer
recount but reproduces the same logic:

> For z=0 layer, the cubes that intersect are: ... 6.
> For z=1 layer, ... 8.
> For z=2 layer, ... 8.
> So the total count is 6 + 8 + 8 = 22. ... Therefore, the correct answer is 22.
> **Final Answer** \boxed{22}

**Injection #2 → R3 commits 19 (correct).** R3 identifies the actual bug —
the criterion for "intersected" was wrong:

> For example, the cube (0,0,0) has all corners with sum from 0 to 3, so no
> intersection. Cube (0,0,1) has corners with sums from 1 to 4, so all
> below 4.5. No intersection.
>
> ... why did I get 22 earlier? Because I was counting cubes based on their
> corners' sums, but the correct condition is based on the lower corner's sum.
> So the correct number is 19.
> **Final Answer** \boxed{19}

**Recovery type.** Multi-stage drill-down. R2 redid the same computation with
the same wrong criterion. Only the second injection pushed the model to
question the *criterion itself*, not just the arithmetic. This is a strong
demonstration of the value of allowing more than one retry.

---

## Summary

| Recovery type | Round of fix | Mechanism | Examples |
|---|---|---|---|
| Constraint misreading | R2 | Re-read the problem | A.1 |
| Last-step inversion | R2 | Recount / pick the right class | A.2 |
| Structural rethink | R2 | Spot a wrong asymptotic / lemma | A.3 |
| Hypothesis search | R3 | Try a different problem instance | B.1 |
| Criterion bug | R3 | Question the rule, not the arithmetic | B.2 |

The Case A traces are the cleanest "epistemic" signal — the model produced a
specific committed answer, recognized something was off given the injection
prompt, and self-corrected. Case B traces are noisier but more interesting:
they show the model failing to fix the bug on the first retry and only
identifying the actual error on the second pass.
