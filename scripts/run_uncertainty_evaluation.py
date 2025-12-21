#!/usr/bin/env python3
"""
Uncertainty-Aware Evaluation Runner for AgentBench.

This script runs uncertainty estimation evaluations on AgentBench tasks
(ToolEmu, DBBench, OS, etc.) using the API-ORCHA-Bench framework.

Usage:
    # Run on ToolEmu results
    python scripts/run_uncertainty_evaluation.py \
        --input outputs/toolemu_gemini/runs.jsonl \
        --task-type toolemu \
        --output outputs/uncertainty_analysis

    # Run on DBBench results
    python scripts/run_uncertainty_evaluation.py \
        --input outputs/dbbench/runs.jsonl \
        --task-type dbbench

    # Run with custom config
    python scripts/run_uncertainty_evaluation.py \
        --input outputs/runs.jsonl \
        --uncertainty-threshold 0.4 \
        --enable-mitigation

    # Live evaluation with agent (requires running controller)
    python scripts/run_uncertainty_evaluation.py \
        --live \
        --agent my-gemini \
        --task toolemu-std \
        --samples 10
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path
script_dir = Path(__file__).parent
repo_root = script_dir.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from agentbench_debug.uncertainty.orchestration_harness import (
    OrchestrationHarness,
    EvaluationConfig,
    evaluate_trajectory,
)
from agentbench_debug.uncertainty.hierarchical import HierarchicalUncertaintyPropagator
from agentbench_debug.uncertainty.calibration import CalibrationMetrics
from agentbench_debug.uncertainty.error_taxonomy import ErrorTaxonomy
from agentbench_debug.uncertainty.scoring import ScoringSystem


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run uncertainty-aware evaluation on AgentBench results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Input/Output
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Path to runs.jsonl file or directory containing results",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs/uncertainty_analysis",
        help="Output directory for analysis results",
    )
    
    # Task configuration
    parser.add_argument(
        "--task-type", "-t",
        type=str,
        choices=["toolemu", "dbbench", "os", "alfworld", "kg", "auto"],
        default="auto",
        help="Task type (auto-detected if not specified)",
    )
    
    # Uncertainty settings
    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=0.35,
        help="Threshold for flagging high uncertainty (default: 0.35)",
    )
    
    # Scoring weights
    parser.add_argument(
        "--progress-weight",
        type=float,
        default=0.4,
        help="Weight for progress score in composite (default: 0.4)",
    )
    parser.add_argument(
        "--completion-weight",
        type=float,
        default=0.3,
        help="Weight for completion score in composite (default: 0.3)",
    )
    parser.add_argument(
        "--interaction-weight",
        type=float,
        default=0.3,
        help="Weight for interaction term in composite (default: 0.3)",
    )
    
    # Mitigation
    parser.add_argument(
        "--enable-mitigation",
        action="store_true",
        help="Enable mitigation strategy analysis",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for mitigation (default: 3)",
    )
    
    # Live evaluation
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live evaluation with agent (requires controller)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="my-gemini",
        help="Agent to use for live evaluation",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="Task name for live evaluation (e.g., toolemu-std)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples for live evaluation",
    )
    parser.add_argument(
        "--controller-url",
        type=str,
        default="http://localhost:5020/api",
        help="Controller URL for live evaluation",
    )
    
    # Output options
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "markdown", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def detect_task_type(file_path: Path) -> str:
    """Auto-detect task type from file path or content."""
    path_str = str(file_path).lower()
    
    if "toolemu" in path_str:
        return "toolemu"
    if "dbbench" in path_str or "db-bench" in path_str:
        return "dbbench"
    if "os" in path_str or "os_interaction" in path_str:
        return "os"
    if "alfworld" in path_str:
        return "alfworld"
    if "kg" in path_str or "knowledge" in path_str:
        return "kg"
    
    # Try to detect from content
    try:
        with open(file_path, "r") as f:
            first_line = f.readline()
            data = json.loads(first_line)
            
            if "Toolkits" in data or "risky" in str(data).lower():
                return "toolemu"
            if "sql" in str(data).lower() or "query" in str(data).lower():
                return "dbbench"
    except Exception:
        pass
    
    return "unknown"


def load_runs(input_path: str) -> List[Dict[str, Any]]:
    """Load runs from file or directory."""
    path = Path(input_path)
    runs = []
    
    if path.is_file():
        if path.suffix == ".jsonl":
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        runs.append(json.loads(line))
        elif path.suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    runs = data
                else:
                    runs = [data]
    
    elif path.is_dir():
        # Look for runs.jsonl files
        for jsonl in path.rglob("runs.jsonl"):
            with open(jsonl, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        runs.append(json.loads(line))
    
    return runs


def extract_steps_from_run(run: Dict[str, Any], task_type: str) -> List[Dict[str, Any]]:
    """Extract step information from a run based on task type."""
    steps = []
    
    # Common extraction
    output = run.get("output", {})
    history = run.get("history", output.get("history", []))
    
    if task_type == "toolemu":
        # ToolEmu format
        for i, item in enumerate(history):
            if isinstance(item, dict):
                steps.append({
                    "action": item.get("tool", item.get("action", "unknown")),
                    "action_type": "tool_call",
                    "messages": item.get("messages", []),
                    "tools": item.get("tools", []),
                    "response": item.get("result", item.get("response")),
                    "tool_calls": item.get("tool_calls", []),
                    "confidence": 1.0 - (0.1 * (i // 3)),  # Simulated decay
                })
    
    elif task_type == "dbbench":
        # DBBench format
        for i, item in enumerate(history):
            if isinstance(item, dict):
                steps.append({
                    "action": "execute_sql",
                    "action_type": "query",
                    "messages": item.get("messages", []),
                    "tools": ["execute_sql", "commit_final_answer"],
                    "response": item.get("result", item.get("response")),
                    "confidence": 0.9 if item.get("success") else 0.5,
                })
    
    elif task_type == "os":
        # OS interaction format
        for i, item in enumerate(history):
            if isinstance(item, dict):
                steps.append({
                    "action": item.get("command", item.get("action", "shell")),
                    "action_type": "system",
                    "messages": item.get("messages", []),
                    "tools": ["bash", "python"],
                    "response": item.get("output", item.get("response")),
                    "confidence": 0.85,
                })
    
    else:
        # Generic extraction
        for i, item in enumerate(history):
            if isinstance(item, dict):
                steps.append({
                    "action": item.get("action", f"step_{i}"),
                    "action_type": "default",
                    "messages": item.get("messages", []),
                    "tools": item.get("tools", []),
                    "response": item.get("response"),
                    "confidence": 0.8,
                })
    
    return steps


def analyze_runs(
    runs: List[Dict[str, Any]],
    task_type: str,
    config: EvaluationConfig,
    verbose: bool = False,
) -> OrchestrationHarness:
    """Analyze runs with uncertainty estimation."""
    harness = OrchestrationHarness(config)
    
    for i, run in enumerate(runs):
        if verbose:
            print(f"Processing run {i + 1}/{len(runs)}...")
        
        task_name = run.get("task", run.get("id", f"run_{i}"))
        success = run.get("success", run.get("output", {}).get("success", False))
        
        # Extract steps
        steps = extract_steps_from_run(run, task_type)
        
        if not steps:
            if verbose:
                print(f"  Skipping run {i + 1}: no steps found")
            continue
        
        # Start run
        harness.start_run(task_name, task_type)
        
        # Record steps
        for step in steps:
            harness.record_step(
                action_name=step["action"],
                action_type=step["action_type"],
                input_messages=step.get("messages", []),
                tools_available=step.get("tools", []),
                response=step.get("response"),
                tool_calls=step.get("tool_calls"),
                confidence=step.get("confidence", 1.0),
            )
        
        # End run
        harness.end_run(success)
    
    return harness


def generate_markdown_report(harness: OrchestrationHarness) -> str:
    """Generate a markdown report of the analysis."""
    metrics = harness.get_aggregate_metrics()
    error_analysis = harness.get_error_analysis()
    uncertainty_analysis = harness.get_uncertainty_analysis()
    
    lines = [
        "# Uncertainty-Aware Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        f"- **Total Runs**: {metrics.get('total_runs', 0)}",
        f"- **Success Rate**: {metrics.get('success_rate', 0):.1%}",
        f"- **Mean Composite Score**: {metrics.get('scores', {}).get('mean_composite', 0):.3f}",
        f"- **Mean ECE**: {metrics.get('calibration', {}).get('mean_ece', 0):.4f}",
        "",
        "## Scoring Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Progress Score (mean) | {metrics.get('scores', {}).get('mean_progress', 0):.3f} |",
        f"| Completion Score (mean) | {metrics.get('scores', {}).get('mean_completion', 0):.3f} |",
        f"| Composite Score (mean) | {metrics.get('scores', {}).get('mean_composite', 0):.3f} |",
        f"| Uncertainty-Adjusted (mean) | {metrics.get('scores', {}).get('mean_uncertainty_adjusted', 0):.3f} |",
        "",
        "## Calibration Metrics",
        "",
        "| Metric | Value | Interpretation |",
        "|--------|-------|----------------|",
    ]
    
    ece = metrics.get('calibration', {}).get('mean_ece', 0)
    ece_interp = "Excellent" if ece < 0.05 else "Good" if ece < 0.10 else "Fair" if ece < 0.15 else "Poor"
    lines.append(f"| Expected Calibration Error (ECE) | {ece:.4f} | {ece_interp} |")
    
    brier = metrics.get('calibration', {}).get('mean_brier', 0)
    brier_interp = "Excellent" if brier < 0.1 else "Good" if brier < 0.2 else "Fair" if brier < 0.3 else "Poor"
    lines.append(f"| Brier Score | {brier:.4f} | {brier_interp} |")
    
    lines.extend([
        "",
        "## Uncertainty Analysis",
        "",
        f"- **Mean Trajectory Uncertainty**: {uncertainty_analysis.get('mean_trajectory_uncertainty', 0):.3f}",
        f"- **Mean Critical Steps per Run**: {uncertainty_analysis.get('mean_critical_steps', 0):.1f}",
        f"- **Aggregated Confidence**: {uncertainty_analysis.get('aggregated_confidence', 0):.3f}",
        "",
        "### Uncertainty Trend Distribution",
        "",
    ])
    
    trends = uncertainty_analysis.get('trend_distribution', {})
    total_trends = sum(trends.values()) or 1
    for trend, count in trends.items():
        lines.append(f"- {trend.capitalize()}: {count} ({count/total_trends:.1%})")
    
    lines.extend([
        "",
        "## Error Analysis",
        "",
        f"- **Total Errors**: {error_analysis.get('errors_by_category', {}) and sum(error_analysis['errors_by_category'].values()) or 0}",
        "",
        "### Errors by Category",
        "",
        "| Category | Count | Percentage |",
        "|----------|-------|------------|",
    ])
    
    by_category = error_analysis.get('errors_by_category', {})
    total_errors = sum(by_category.values()) or 1
    for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
        lines.append(f"| {cat} | {count} | {count/total_errors:.1%} |")
    
    lines.extend([
        "",
        "### Top Error Types",
        "",
    ])
    
    top_errors = error_analysis.get('top_10_errors', [])
    if top_errors:
        lines.append("| Error Type | Count |")
        lines.append("|------------|-------|")
        for error_type, count in top_errors[:10]:
            lines.append(f"| {error_type} | {count} |")
    else:
        lines.append("No errors recorded.")
    
    lines.extend([
        "",
        "## Resource Usage",
        "",
        f"- **Mean Latency**: {metrics.get('latency', {}).get('mean_ms', 0):.1f} ms",
        f"- **Total Tokens**: {metrics.get('tokens', {}).get('total', 0):,}",
        f"- **Mean Tokens per Run**: {metrics.get('tokens', {}).get('mean_per_run', 0):.0f}",
        "",
        "---",
        "",
        "*Report generated by API-ORCHA-Bench Uncertainty Evaluation Framework*",
    ])
    
    return "\n".join(lines)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create config
    config = EvaluationConfig(
        uncertainty_threshold=args.uncertainty_threshold,
        enable_hierarchical=True,
        enable_calibration=True,
        progress_weight=args.progress_weight,
        completion_weight=args.completion_weight,
        interaction_weight=args.interaction_weight,
        enable_mitigation=args.enable_mitigation,
        max_retries=args.max_retries,
        output_dir=args.output,
    )
    
    # Handle live evaluation
    if args.live:
        print("Live evaluation not yet implemented.")
        print("Use --input to analyze pre-recorded runs.")
        sys.exit(1)
    
    # Validate input
    if not args.input:
        print("Error: --input is required for offline analysis")
        sys.exit(1)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # Detect task type
    task_type = args.task_type
    if task_type == "auto":
        task_type = detect_task_type(input_path)
        if args.verbose:
            print(f"Auto-detected task type: {task_type}")
    
    # Load runs
    print(f"Loading runs from {input_path}...")
    runs = load_runs(str(input_path))
    print(f"Loaded {len(runs)} runs")
    
    if not runs:
        print("No runs found. Check your input path.")
        sys.exit(1)
    
    # Analyze runs
    print("Analyzing runs with uncertainty estimation...")
    harness = analyze_runs(runs, task_type, config, args.verbose)
    
    # Get results
    metrics = harness.get_aggregate_metrics()
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Runs: {metrics.get('total_runs', 0)}")
    print(f"Success Rate: {metrics.get('success_rate', 0):.1%}")
    print(f"Mean Composite Score: {metrics.get('scores', {}).get('mean_composite', 0):.3f}")
    print(f"Mean ECE: {metrics.get('calibration', {}).get('mean_ece', 0):.4f}")
    print(f"Mean Brier Score: {metrics.get('calibration', {}).get('mean_brier', 0):.4f}")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.format in ["json", "both"]:
        # Save JSON
        json_path = output_dir / f"analysis_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump({
                "config": {
                    "task_type": task_type,
                    "uncertainty_threshold": config.uncertainty_threshold,
                    "weights": {
                        "progress": config.progress_weight,
                        "completion": config.completion_weight,
                        "interaction": config.interaction_weight,
                    },
                },
                "metrics": metrics,
                "error_analysis": harness.get_error_analysis(),
                "uncertainty_analysis": harness.get_uncertainty_analysis(),
            }, f, indent=2)
        print(f"\nJSON report saved to: {json_path}")
    
    if args.format in ["markdown", "both"]:
        # Save Markdown
        md_path = output_dir / f"report_{timestamp}.md"
        with open(md_path, "w") as f:
            f.write(generate_markdown_report(harness))
        print(f"Markdown report saved to: {md_path}")
    
    # Also save full harness results
    harness.save_results(str(output_dir))
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()

