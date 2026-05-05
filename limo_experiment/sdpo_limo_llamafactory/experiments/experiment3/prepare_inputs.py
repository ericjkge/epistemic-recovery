"""
Build per-model JSONL inputs for the MI-peak calc on AIME25 problem 23
(= 2025 AIME II Problem 9).

Problem 23 is one of the three "LoRA-only solved" problems flagged in
experiment2/NOTES.md (LoRA: 4/8 correct, baseline SDPO: 0/8). We pick the
first correct LoRA generation and the first wrong baseline generation as
matched traces for MI comparison.

Each record includes a `solution` field — a worked solution for use as the
GT representation when running calc.py with --use_solution. AoPS blocks
programmatic fetches; the solution below is hand-derived from the same
mathematics laid out in experiment1/eval_results/epistemic_case_studies.md.
"""

import json
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
EVAL_DIR = EXP_DIR.parent / "experiment2/eval_results"
OUT_DIR = EXP_DIR / "inputs"
OUT_DIR.mkdir(exist_ok=True)

LORA_FILE = EVAL_DIR / "lora_sdpo_plus_limo_aime25.json"
BASE_FILE = EVAL_DIR / "baseline_sdpo_think_step100_aime25.json"
PROBLEM_IDX = 23

SOLUTION = (
    "We have f(x) = sin(7π · sin(5x)) on the open interval 0 < x < 2π. "
    "Setting f(x) = 0 requires 7π · sin(5x) = kπ for some integer k, "
    "equivalently sin(5x) = k/7 with k ∈ {-7, -6, ..., 6, 7}. "
    "Substitute θ = 5x; then 0 < θ < 10π covers exactly five periods of sin. "
    "For c ∈ (-1, 0) ∪ (0, 1) the equation sin(θ) = c has 2 solutions per "
    "period, so 5 × 2 = 10 solutions on the open interval. The 12 nonzero "
    "values k ∈ {-6, ..., -1, 1, ..., 6} each give 10 solutions, contributing "
    "12 × 10 = 120. For c = ±1 (i.e. k = ±7), sin(θ) = ±1 has 1 solution per "
    "period, so 5 each, contributing 2 × 5 = 10. For c = 0 (k = 0), the zeros "
    "of sin(θ) on 0 < θ < 10π are θ = π, 2π, ..., 9π — exactly 9 solutions, "
    "since θ = 10π lies on the excluded endpoint of the open interval. "
    "Therefore n = 120 + 10 + 9 = 139. "
    "Tangency requires f(x) = 0 and f'(x) = 0. Differentiating gives "
    "f'(x) = 35π · cos(7π · sin(5x)) · cos(5x). Combined with f(x) = 0, "
    "we have two cases. Case A: cos(7π · sin(5x)) = 0, so 7π · sin(5x) = "
    "π/2 + mπ, giving sin(5x) = (2m + 1)/14. But f(x) = 0 forces sin(5x) = "
    "k/7 = 2k/14, requiring 2k = 2m + 1, impossible (even = odd). No "
    "tangencies arise here. Case B: cos(5x) = 0, so sin(5x) = ±1, forcing "
    "k = ±7. All 10 zeros at k = ±7 are tangent points. Therefore t = 10. "
    "Finally n + t = 139 + 10 = 149."
)

lora = json.loads(LORA_FILE.read_text())
base = json.loads(BASE_FILE.read_text())

lora_r = lora["results"][PROBLEM_IDX]
base_r = base["results"][PROBLEM_IDX]

lora_gen_idx = next(i for i, c in enumerate(lora_r["correctness"]) if c)
base_gen_idx = next(i for i, c in enumerate(base_r["correctness"]) if not c)

print(f"LoRA: gen[{lora_gen_idx}] -> {lora_r['extracted_answers'][lora_gen_idx]} (gt={lora_r['ground_truth']})")
print(f"Base: gen[{base_gen_idx}] -> {base_r['extracted_answers'][base_gen_idx]} (gt={base_r['ground_truth']})")


def write_record(path: Path, result: dict, gen_idx: int):
    rec = {
        "problem_id": result["problem_id"],
        "question": result["question"],
        "gold_answer": str(result["ground_truth"]),
        "solution": SOLUTION,
        "generated_responses": [result["generations"][gen_idx]],
    }
    path.write_text(json.dumps(rec) + "\n")
    print(f"Wrote {path.name} ({len(result['generations'][gen_idx])} chars trajectory, {len(SOLUTION)} chars solution)")


write_record(OUT_DIR / "aime23_lora_correct.jsonl", lora_r, lora_gen_idx)
write_record(OUT_DIR / "aime23_base_wrong.jsonl", base_r, base_gen_idx)
