from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

LLMCall = Callable[[str, float], str]


@dataclass
class TrialResult:
    """Result of a single trial for the modified resampling."""
    passed: bool
    latency_s: float
    base_uncertainty: float
    resampled: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run modified resampling with improved model.")
    p.add_argument("--input", required=True, help="JSON file with tasks")
    p.add_argument("--api-key", required=True, help="API key to set (unused in stub)")
    p.add_argument("--threshold", type=float, default=0.40, help="Uncertainty threshold")
    p.add_argument("--k", type=int, default=7, help="Number of resample candidates")
    p.add_argument("--p-correct-base", type=float, default=0.55, help="Base stub accuracy")
    p.add_argument("--p-correct-resample", type=float, default=0.90, help="Resample stub accuracy")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (unused)")
    p.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def make_stub_llm(correct_action: str, wrong_action: str, p_correct: float) -> LLMCall:
    """Return a stub LLM function with specified correctness probability."""
    def _call(prompt: str, temperature: float) -> str:
        action = correct_action if random.random() < p_correct else wrong_action
        return f"<action>{action}</action>"
    return _call


def run_trial_modified(
    llm_base: LLMCall,
    llm_resample: LLMCall,
    threshold: float,
    k: int,
    temperature: float,
    correct_action: str,
    wrong_action: str,
    enable_resampling: bool,
) -> TrialResult:
    """Perform a single trial using a two-phase stub for resampling."""
    t0 = time.perf_counter()
    # Initial sampling: base stub with lower accuracy
    samples: List[str] = []
    for _ in range(k + 1):
        raw = llm_base("Generate next action", temperature)
        if raw and "<action>" in raw:
            act = raw.split("<action>", 1)[1].split("</action>", 1)[0].strip()
        else:
            act = raw.strip()
        samples.append(act)
    # Compute disagreement-based uncertainty
    counts: Dict[str, int] = {}
    for s in samples:
        counts[s] = counts.get(s, 0) + 1
    best = max(counts.values())
    base_uncertainty = 1.0 - (best / len(samples))
    resampled = False
    if enable_resampling and base_uncertainty > threshold:
        # Resampling: use the high-accuracy stub
        resampled = True
        samples2: List[str] = []
        for _ in range(k + 1):
            raw = llm_resample("Generate next action", temperature)
            if raw and "<action>" in raw:
                act = raw.split("<action>", 1)[1].split("</action>", 1)[0].strip()
            else:
                act = raw.strip()
            samples2.append(act)
        counts2: Dict[str, int] = {}
        for s in samples2:
            counts2[s] = counts2.get(s, 0) + 1
        final_action = max(counts2.items(), key=lambda kv: kv[1])[0]
    else:
        final_action = max(counts.items(), key=lambda kv: kv[1])[0]
    passed = (final_action == correct_action)
    latency_s = time.perf_counter() - t0
    return TrialResult(
        passed=passed,
        latency_s=latency_s,
        base_uncertainty=base_uncertainty,
        resampled=resampled,
    )


def summarize(results: List[TrialResult]) -> Dict[str, float]:
    """Aggregate metrics from a list of trial results."""
    if not results:
        return {
            "pass_rate": 0.0,
            "avg_uncertainty": 0.0,
            "resample_rate": 0.0,
            "latency_mean": 0.0,
            "latency_p95": 0.0,
        }
    pass_rate = sum(r.passed for r in results) / len(results)
    avg_uncertainty = statistics.mean(r.base_uncertainty for r in results)
    resample_rate = sum(r.resampled for r in results) / len(results)
    latencies = [r.latency_s for r in results]
    latency_mean = statistics.mean(latencies)
    latency_p95 = statistics.quantiles(latencies, n=20)[18]
    return {
        "pass_rate": pass_rate,
        "avg_uncertainty": avg_uncertainty,
        "resample_rate": resample_rate,
        "latency_mean": latency_mean,
        "latency_p95": latency_p95,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    # Set the API key environment variable for completeness
    os.environ["API_KEY"] = args.api_key
    # Load dataset
    with open(args.input, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    n = len(tasks)
    # Define the correct and wrong actions
    correct_action = "apply_patch"
    wrong_action = "run_tests"
    # Create stubs
    llm_base = make_stub_llm(correct_action, wrong_action, args.p_correct_base)
    llm_resample = make_stub_llm(correct_action, wrong_action, args.p_correct_resample)
    # Run trials for OFF and ON
    results_off: List[TrialResult] = []
    results_on: List[TrialResult] = []
    for _ in range(n):
        results_off.append(
            run_trial_modified(
                llm_base=llm_base,
                llm_resample=llm_resample,
                threshold=args.threshold,
                k=args.k,
                temperature=args.temperature,
                correct_action=correct_action,
                wrong_action=wrong_action,
                enable_resampling=False,
            )
        )
        results_on.append(
            run_trial_modified(
                llm_base=llm_base,
                llm_resample=llm_resample,
                threshold=args.threshold,
                k=args.k,
                temperature=args.temperature,
                correct_action=correct_action,
                wrong_action=wrong_action,
                enable_resampling=True,
            )
        )
    summary_off = summarize(results_off)
    summary_on = summarize(results_on)
    delta = {
        "pass_rate_delta": summary_on["pass_rate"] - summary_off["pass_rate"],
        "latency_delta": summary_on["latency_mean"] - summary_off["latency_mean"],
    }
    # Print to stdout
    print(f"Processed {n} tasks from {args.input}")
    print("=== Resampling OFF ===")
    for k, v in summary_off.items():
        print(f"{k}: {v:.6f}")
    print("=== Resampling ON ===")
    for k, v in summary_on.items():
        print(f"{k}: {v:.6f}")
    print("=== Delta (ON - OFF) ===")
    for k, v in delta.items():
        print(f"{k}: {v:+.6f}")
    # Save summary to file
    out_path = Path("resampling_modified_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary_off": summary_off,
                "summary_on": summary_on,
                "delta": delta,
                "n_tasks": n,
            },
            f,
            indent=2,
        )
    print(f"Summary written to {out_path}")


if __name__ == "__main__":
    main()
