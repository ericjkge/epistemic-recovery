"""Epistemic alignment metrics following Figure 9 of Kim et al. (arXiv:2603.15500).

Provides:
  build_epistemic_token_ids(tokenizer)   — tensor of all epistemic token IDs
  build_diagnostic_token_map(tokenizer)  — per-token tensors (wait, alternatively, …)
  load_probe_set(path, n, seed)          — sample LIMO traces for probing
  prepare_probe_set(tokenizer, traces)   — pre-tokenize once; pass `prepared` to avoid
                                           re-tokenizing on every evaluation call
  compute_epistemic_metrics(model, tokenizer, probe_traces, epistemic_token_ids, …)
"""

import json
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


EPISTEMIC_TOKEN_STRINGS = [
    "wait", "hmm", "perhaps", "maybe", "actually",
    "alternatively", "seems", "might", "likely", "check",
]

# The four most diagnostic per Kim et al. — used for per-token log-prob breakdown
DIAGNOSTIC_TOKENS = ["wait", "alternatively", "maybe", "hmm"]

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


# ── Token ID resolution ───────────────────────────────────────────────────────

def build_epistemic_token_ids(tokenizer) -> torch.Tensor:
    """Resolve all spacing/case variants of the 10 epistemic tokens to a tensor of IDs.

    Variants checked: word, Word, " word", " Word".
    Leading-space variants are critical for BPE tokenizers (e.g. Qwen3).
    Only single-token encodings are included; multi-token encodings are skipped.
    """
    token_ids: set[int] = set()
    for word in EPISTEMIC_TOKEN_STRINGS:
        for variant in (word, word.capitalize(), f" {word}", f" {word.capitalize()}"):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.add(ids[0])
    return torch.tensor(sorted(token_ids), dtype=torch.long)


def build_diagnostic_token_map(tokenizer) -> dict[str, torch.Tensor]:
    """Map each diagnostic token name → tensor of its token IDs (all single-token variants).

    Used for per-token log-prob breakdown and Figure 9 B/C density plots.
    """
    result: dict[str, torch.Tensor] = {}
    for word in DIAGNOSTIC_TOKENS:
        ids: set[int] = set()
        for variant in (word, word.capitalize(), f" {word}", f" {word.capitalize()}"):
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if len(encoded) == 1:
                ids.add(encoded[0])
        if ids:
            result[word] = torch.tensor(sorted(ids), dtype=torch.long)
    return result


# ── Probe set loading ─────────────────────────────────────────────────────────

def load_probe_set(dataset_path: str, n_probes: int = 50, seed: int = 42) -> list[dict]:
    """Sample n_probes traces from a LlamaFactory-format JSON dataset.

    Fixed seed ensures the same probe set across all evaluation calls.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(seed)
    if len(data) > n_probes:
        data = rng.sample(data, n_probes)
    return data


def _build_prompt(tokenizer, record: dict) -> str:
    messages = []
    sys_msg = record.get("system", SYSTEM_PROMPT)
    if sys_msg and sys_msg.strip():
        messages.append({"role": "system", "content": sys_msg})
    user_content = record.get("instruction", record.get("question", ""))
    inp = record.get("input", "")
    if inp and inp.strip():
        user_content = f"{user_content}\n{inp}"
    messages.append({"role": "user", "content": user_content})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prepare_probe_set(
    tokenizer,
    probe_traces: list[dict],
    max_len: int = 16384,
) -> list[dict]:
    """Pre-tokenize probe traces so tokenization is not repeated every evaluation call.

    Returns a list of {'input_ids': list[int], 'prompt_len': int}.
    """
    prepared = []
    for record in probe_traces:
        prompt_text = _build_prompt(tokenizer, record)
        output_text = record.get("output", record.get("solution", ""))

        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        prompt_len = int(prompt_ids.shape[1])

        full_ids = tokenizer(
            prompt_text + output_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )["input_ids"].squeeze(0)

        if full_ids.shape[0] < prompt_len + 2:
            continue

        prepared.append({"input_ids": full_ids.tolist(), "prompt_len": prompt_len})
    return prepared


# ── Core metric computation ───────────────────────────────────────────────────

@torch.no_grad()
def compute_epistemic_metrics(
    model,
    tokenizer,
    probe_traces: list[dict],
    epistemic_token_ids: torch.Tensor,
    diagnostic_token_map: Optional[dict[str, torch.Tensor]] = None,
    subsample_size: int = 50_000,
    prepared: Optional[list[dict]] = None,
) -> dict:
    """Teacher-force each probe trace through model and return epistemic alignment metrics.

    If `prepared` (from prepare_probe_set) is provided, it is used directly and
    tokenization is skipped — strongly recommended for repeated evaluation calls.

    Scalar metrics (log to wandb as epistemic/<key>):
      epistemic_alignment_gap    mean logprob[all] − mean logprob[epistemic]
                                 smaller gap → epistemic tokens more in-distribution
      epistemic_entropy_ratio    mean entropy[epistemic] / mean entropy[all]
                                 >1 → epistemic tokens sit at genuine decision points
      trace_mean_logprob         mean logprob across all answer tokens + traces
      epistemic_mean_logprob     mean logprob restricted to epistemic positions
      all_tokens_mean_logprob    mean logprob at non-epistemic positions
      mean_logprob_{token}       per-diagnostic-token mean logprob

    Array outputs (for Figure 9 plotting):
      per_trace_avg_logprobs      list[float], one per trace (for CDF plot)
      all_token_logprobs_sample   subsampled list of all token logprobs
      epistemic_token_logprobs    all logprobs at epistemic positions
      all_token_entropies_sample  subsampled list of all token entropies
      epistemic_token_entropies   all entropies at epistemic positions
      {token}_token_logprobs      per-diagnostic-token logprob arrays
      {token}_token_entropies     per-diagnostic-token entropy arrays
    """
    was_training = model.training
    model.eval()

    if diagnostic_token_map is None:
        diagnostic_token_map = {}

    device = next(model.parameters()).device
    epistemic_ids = epistemic_token_ids.to(device)
    diag_ids = {k: v.to(device) for k, v in diagnostic_token_map.items()}

    all_lp: list[float] = []
    non_ep_lp: list[float] = []
    ep_lp: list[float] = []
    all_ent: list[float] = []
    ep_ent: list[float] = []
    per_trace_avg: list[float] = []
    diag_lp: dict[str, list[float]] = {k: [] for k in diagnostic_token_map}
    diag_ent: dict[str, list[float]] = {k: [] for k in diagnostic_token_map}

    def _process(input_ids_list: list[int], prompt_len: int) -> None:
        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        logits = model(input_ids=input_ids.unsqueeze(0)).logits  # [1, L, V]

        lp_dist = F.log_softmax(logits.squeeze(0), dim=-1)      # [L, V]
        pr_dist = lp_dist.exp()

        # Shift: position i predicts token i+1
        target_ids = input_ids[1:]           # [L-1]
        lp_shifted = lp_dist[:-1]            # [L-1, V]
        pr_shifted = pr_dist[:-1]            # [L-1, V]

        token_lp = lp_shifted.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)  # [L-1]
        entropy = -(pr_shifted * lp_shifted).sum(-1)                            # [L-1]

        ans_start = prompt_len - 1
        if ans_start >= token_lp.shape[0]:
            return

        ans_ids = target_ids[ans_start:]
        ans_lp = token_lp[ans_start:]
        ans_ent = entropy[ans_start:]

        if ans_lp.numel() == 0:
            return

        is_ep = torch.isin(ans_ids, epistemic_ids)

        per_trace_avg.append(float(ans_lp.mean()))
        all_lp.extend(ans_lp.cpu().float().tolist())
        all_ent.extend(ans_ent.cpu().float().tolist())

        if is_ep.any():
            ep_lp.extend(ans_lp[is_ep].cpu().float().tolist())
            ep_ent.extend(ans_ent[is_ep].cpu().float().tolist())
        if (~is_ep).any():
            non_ep_lp.extend(ans_lp[~is_ep].cpu().float().tolist())

        for token_name, d_ids in diag_ids.items():
            mask = torch.isin(ans_ids, d_ids)
            if mask.any():
                diag_lp[token_name].extend(ans_lp[mask].cpu().float().tolist())
                diag_ent[token_name].extend(ans_ent[mask].cpu().float().tolist())

    if prepared is not None:
        for item in prepared:
            _process(item["input_ids"], item["prompt_len"])
    else:
        for record in probe_traces:
            prompt_text = _build_prompt(tokenizer, record)
            output_text = record.get("output", record.get("solution", ""))
            prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
            prompt_len = int(prompt_ids.shape[1])
            full_ids = tokenizer(
                prompt_text + output_text, return_tensors="pt", add_special_tokens=False,
            )["input_ids"].squeeze(0).tolist()
            if len(full_ids) < prompt_len + 2:
                continue
            _process(full_ids, prompt_len)

    if was_training:
        model.train()

    def safe_mean(lst: list) -> float:
        return float(np.mean(lst)) if lst else float("nan")

    rng_np = np.random.default_rng(42)

    def subsample(lst: list, n: int) -> list:
        arr = np.array(lst, dtype=np.float32)
        if len(arr) > n:
            idx = np.sort(rng_np.choice(len(arr), size=n, replace=False))
            return arr[idx].tolist()
        return arr.tolist()

    trace_mean = safe_mean(all_lp)
    ep_mean = safe_mean(ep_lp)
    non_ep_mean = safe_mean(non_ep_lp)
    all_ent_mean = safe_mean(all_ent)
    ep_ent_mean = safe_mean(ep_ent)

    def _gap(a: float, b: float) -> float:
        return float("nan") if (a != a or b != b) else a - b

    def _ratio(num: float, denom: float) -> float:
        return float("nan") if (num != num or denom != denom or denom == 0.0) else num / denom

    scalars: dict = {
        "epistemic_alignment_gap": _gap(trace_mean, ep_mean),
        "epistemic_entropy_ratio": _ratio(ep_ent_mean, all_ent_mean),
        "trace_mean_logprob": trace_mean,
        "epistemic_mean_logprob": ep_mean,
        "all_tokens_mean_logprob": non_ep_mean,
    }
    for token_name in diagnostic_token_map:
        scalars[f"mean_logprob_{token_name}"] = safe_mean(diag_lp[token_name])

    arrays: dict = {
        "per_trace_avg_logprobs": per_trace_avg,
        "all_token_logprobs_sample": subsample(all_lp, subsample_size),
        "epistemic_token_logprobs": ep_lp,
        "all_token_entropies_sample": subsample(all_ent, subsample_size),
        "epistemic_token_entropies": ep_ent,
    }
    for token_name in diagnostic_token_map:
        arrays[f"{token_name}_token_logprobs"] = diag_lp[token_name]
        arrays[f"{token_name}_token_entropies"] = diag_ent[token_name]

    return {**scalars, **arrays}
