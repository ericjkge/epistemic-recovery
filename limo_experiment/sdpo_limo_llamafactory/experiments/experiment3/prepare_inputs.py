"""
Build per-model JSONL inputs for the MI-peak calc on AIME25 problem 23.

Problem 23 is one of the three "LoRA-only solved" problems flagged in
experiment2/NOTES.md (LoRA: 4/8 correct, baseline SDPO: 0/8). We pick the
first correct LoRA generation and the first wrong baseline generation as
matched traces for MI comparison.
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
        "generated_responses": [result["generations"][gen_idx]],
    }
    path.write_text(json.dumps(rec) + "\n")
    print(f"Wrote {path.name} ({len(result['generations'][gen_idx])} chars)")


write_record(OUT_DIR / "aime23_lora_correct.jsonl", lora_r, lora_gen_idx)
write_record(OUT_DIR / "aime23_base_wrong.jsonl", base_r, base_gen_idx)
