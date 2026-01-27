#!/usr/bin/env python3
"""
Post-hoc analysis script for SWE-bench and AgentBench results.

This script analyzes completed runs from the outputs directory and computes:
- Calibration metrics (ECE, Brier, Spearman correlation)
- Task outcome analysis
- Confidence distribution by success/failure

Usage:
    python scripts/analyze_swebench_results.py outputs/2026-01-26-14-28-48
    python scripts/analyze_swebench_results.py --latest
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agentbench_debug.uncertainty.calibration import (
    CalibrationMetrics,
    CalibrationResult,
    OutcomeAnalysis,
)


@dataclass
class RunAnalysis:
    """Analysis of a single run."""
    agent: str
    task: str
    index: int
    success: bool
    n_steps: int
    mean_confidence: float
    min_confidence: float
    max_confidence: float
    trajectory_uncertainty: float


def find_latest_output(base_path: str = "outputs") -> Optional[str]:
    """Find the most recent output directory."""
    if not os.path.exists(base_path):
        return None
    
    dirs = []
    for d in os.listdir(base_path):
        full_path = os.path.join(base_path, d)
        if os.path.isdir(full_path):
            dirs.append((os.path.getmtime(full_path), full_path))
    
    if not dirs:
        return None
    
    dirs.sort(reverse=True)
    return dirs[0][1]


def load_uncertainty_analysis(output_dir: str) -> Optional[Dict[str, Any]]:
    """Load existing uncertainty_analysis.json if it exists."""
    analysis_path = os.path.join(output_dir, "uncertainty_analysis.json")
    if os.path.exists(analysis_path):
        with open(analysis_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def print_calibration_report(
    analyses: List[RunAnalysis],
    calibration: CalibrationResult,
    outcome: OutcomeAnalysis,
):
    """Print a detailed calibration report."""
    print("\n" + "=" * 70)
    print("                    UNCERTAINTY CALIBRATION REPORT")
    print("=" * 70)
    
    # Summary statistics
    print("\n[1] SUMMARY STATISTICS")
    print("-" * 50)
    print(f"  Total runs analyzed:        {len(analyses)}")
    print(f"  Successful runs:            {outcome.successful_tasks} ({outcome.success_rate:.1%})")
    print(f"  Failed runs:                {outcome.failed_tasks}")
    
    # Confidence statistics
    all_confs = [a.mean_confidence for a in analyses]
    print(f"\n  Mean confidence (overall):  {sum(all_confs)/len(all_confs):.3f}")
    print(f"  Mean confidence (success):  {outcome.mean_confidence_success:.3f}")
    print(f"  Mean confidence (failure):  {outcome.mean_confidence_failure:.3f}")
    print(f"  Confidence gap:             {outcome.confidence_gap:+.3f}")
    
    # Calibration metrics
    print("\n[2] CALIBRATION METRICS")
    print("-" * 50)
    print(f"  ECE (Expected Calibration Error):  {calibration.ece:.4f}")
    
    if calibration.ece < 0.05:
        print("    -> Well-calibrated: confidence closely matches success rate")
    elif calibration.ece < 0.15:
        print("    -> Moderately calibrated: some gap between confidence and success")
    else:
        print("    -> Poorly calibrated: large discrepancy")
    
    print(f"\n  MCE (Maximum Calibration Error):   {calibration.mce:.4f}")
    print(f"  Brier Score:                       {calibration.brier_score:.4f}")
    
    # Correlation metrics
    print("\n[3] CORRELATION & DISCRIMINATION")
    print("-" * 50)
    print(f"  Spearman rho:  {calibration.spearman_rho:.4f}")
    
    if calibration.spearman_rho > 0.3:
        print("    -> Good: higher confidence correlates with success")
    elif calibration.spearman_rho > 0.1:
        print("    -> Weak positive correlation")
    elif calibration.spearman_rho > -0.1:
        print("    -> No meaningful correlation")
    else:
        print("    -> INVERSE correlation: confident when failing (problematic)")
    
    print(f"\n  AUROC:         {calibration.auroc:.4f}")
    
    if calibration.auroc > 0.7:
        print("    -> Good discrimination: can distinguish success from failure")
    elif calibration.auroc > 0.55:
        print("    -> Weak discrimination")
    else:
        print("    -> Poor discrimination: no better than random")
    
    # Over/underconfidence
    print("\n[4] CONFIDENCE ANALYSIS")
    print("-" * 50)
    print(f"  Overconfidence rate:   {outcome.overconfidence_rate:.1%}")
    print(f"    (% of failures with confidence >= 0.7)")
    print(f"  Underconfidence rate:  {outcome.underconfidence_rate:.1%}")
    print(f"    (% of successes with confidence < 0.5)")
    
    # Reliability diagram (text-based)
    print("\n[5] RELIABILITY DIAGRAM (text)")
    print("-" * 50)
    print("  Bin    | Confidence | Accuracy | Count | Gap")
    print("  -------|------------|----------|-------|-----")
    
    for i in range(calibration.n_bins):
        conf = calibration.bin_confidences[i]
        acc = calibration.bin_accuracies[i]
        count = calibration.bin_counts[i]
        gap = acc - conf
        
        if count > 0:
            bar = "#" * min(int(count / 2), 20)
            print(f"  {i:2d}-{i+1:2d}0% | {conf:.3f}      | {acc:.3f}    | {count:5d} | {gap:+.3f} {bar}")
    
    # Interpretation
    print("\n[6] INTERPRETATION")
    print("-" * 50)
    
    # Check if uniform confidence (indicates need for better signal)
    unique_confs = set(a.mean_confidence for a in analyses)
    if len(unique_confs) <= 2:
        print("  WARNING: Confidence values are nearly uniform")
        print("    This suggests the confidence signal is not informative.")
        print("    Possible causes:")
        print("    - finish_reason:tool_calls returns constant 0.80")
        print("    - Logprobs not available for function calling")
        print("    - No semantic hedging detected in agent responses")
        print("\n  RECOMMENDATION: Enable semantic analysis of tool arguments")
        print("    or implement resampling-based uncertainty estimation.")
    else:
        if outcome.confidence_gap > 0.05:
            print("  + Agent confidence differentiates success from failure")
        else:
            print("  - Agent confidence doesn't differentiate outcomes well")
        
        if calibration.spearman_rho > 0.2:
            print("  + Confidence positively correlates with success")
        elif calibration.spearman_rho < -0.1:
            print("  ! Confidence NEGATIVELY correlates with success (problematic)")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze SWE-bench/AgentBench results")
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Path to output directory (or --latest for most recent)"
    )
    parser.add_argument(
        "--latest", "-l",
        action="store_true",
        help="Use the most recent output directory"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.latest or args.output_dir is None:
        output_dir = find_latest_output()
        if output_dir is None:
            print("Error: No output directories found")
            sys.exit(1)
        print(f"Using latest output: {output_dir}")
    else:
        output_dir = args.output_dir
    
    if not os.path.exists(output_dir):
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)
    
    # Load existing uncertainty analysis
    existing = load_uncertainty_analysis(output_dir)
    
    if existing and "runs" in existing:
        print(f"Loaded uncertainty_analysis.json with {len(existing['runs'])} runs")
        
        # Convert to RunAnalysis objects
        analyses = []
        for run in existing["runs"]:
            analyses.append(RunAnalysis(
                agent=run.get("agent", "unknown"),
                task=run.get("task", "unknown"),
                index=run.get("index", -1),
                success=run.get("success", False),
                n_steps=run.get("n_steps", 0),
                mean_confidence=run.get("mean_confidence", 0.7),
                min_confidence=run.get("min_confidence", 0.7),
                max_confidence=run.get("max_confidence", 0.7),
                trajectory_uncertainty=1 - run.get("mean_confidence", 0.7),
            ))
        
        if not analyses:
            print("Error: No runs found in analysis file")
            sys.exit(1)
        
        # Compute calibration metrics
        confidences = [a.mean_confidence for a in analyses]
        outcomes = [a.success for a in analyses]
        
        calibration_metrics = CalibrationMetrics(n_bins=10)
        calibration_result = calibration_metrics.compute(confidences, outcomes)
        outcome_analysis = calibration_metrics.analyze_outcomes(confidences, outcomes)
        
        if args.json:
            # Output as JSON
            result = {
                "output_dir": output_dir,
                "n_runs": len(analyses),
                "calibration": {
                    "ece": calibration_result.ece,
                    "mce": calibration_result.mce,
                    "brier_score": calibration_result.brier_score,
                    "spearman_rho": calibration_result.spearman_rho,
                    "auroc": calibration_result.auroc,
                },
                "outcome_analysis": {
                    "success_rate": outcome_analysis.success_rate,
                    "mean_confidence_success": outcome_analysis.mean_confidence_success,
                    "mean_confidence_failure": outcome_analysis.mean_confidence_failure,
                    "confidence_gap": outcome_analysis.confidence_gap,
                    "overconfidence_rate": outcome_analysis.overconfidence_rate,
                },
            }
            print(json.dumps(result, indent=2))
        else:
            # Print detailed report
            print_calibration_report(analyses, calibration_result, outcome_analysis)
    else:
        print("Error: No uncertainty_analysis.json found or invalid format")
        print("Run the benchmark with uncertainty tracking first:")
        print("  python -m src.uncertainty_assigner --config <config>.yaml")
        sys.exit(1)


if __name__ == "__main__":
    main()
