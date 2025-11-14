"""Run a lightweight demo of the uncertainty quantifier for ALFWorld steps."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add src/ to Python path so we can import agentbench_debug
script_dir = Path(__file__).parent
repo_root = script_dir.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from agentbench_debug.fine_grained.analyzer import FineGrainedAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to an ALFWorld run JSON file")
    parser.add_argument("--output", required=True, help="Where to write the augmented JSON output")
    parser.add_argument("--threshold", type=float, default=0.35, help="Uncertainty threshold")
    parser.add_argument("--k", type=int, default=3, help="Number of regeneration samples")
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature for regeneration"
    )
    parser.add_argument(
        "--no-self-correction",
        action="store_true",
        help="Disable the optional self-correction step",
    )
    return parser.parse_args()


def load_run(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def make_stub_llm(base_action: str):
    base_action = base_action.strip() or "look around"
    vocab = [
        "inspect",
        "examine",
        "observe",
        "check",
        "explore",
        "search",
    ]

    def _call(prompt: str, temperature: float) -> str:
        tokens = base_action.split()
        if random.random() < min(max(temperature, 0.0), 1.0) and tokens:
            tokens = tokens.copy()
            random.shuffle(tokens)
        synonym = random.choice(vocab)
        candidate = " ".join(tokens) or base_action
        variant = f"{synonym} - {candidate}" if synonym not in base_action else candidate
        return f"<action>{variant}</action>"

    return _call


def main() -> None:
    args = parse_args()
    data = load_run(Path(args.input))

    task_description = (
        data.get("task_description")
        or data.get("task")
        or data.get("task_text")
        or ""
    )
    steps: List[Dict[str, Any]] = data.get("steps") or []

    analyzer = FineGrainedAnalyzer(
        threshold=args.threshold,
        k=args.k,
        temperature=args.temperature,
        enable_self_correction=not args.no_self_correction,
        track_history=False,
    )

    augmented_steps: List[Dict[str, Any]] = []

    for idx, step in enumerate(steps):
        modules = step.get("modules") or {"action": step.get("action", "")}
        base_action = modules.get("action", "")
        analyzer.set_call_fn(make_stub_llm(base_action))

        summary = analyzer.analyze_step(
            step_idx=step.get("idx", idx),
            task_description=task_description,
            context_text=step.get("context") or step.get("context_text", ""),
            modules=modules,
        )
        augmented = {**step, **summary}
        augmented_steps.append(augmented)

    result: Dict[str, Any] = {
        "task": data.get("task") or data.get("task_id"),
        "task_description": task_description,
        "steps": augmented_steps,
    }

    analyzer.attach_totals(result, augmented_steps)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

