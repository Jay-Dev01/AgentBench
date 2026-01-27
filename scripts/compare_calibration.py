#!/usr/bin/env python3
"""
Compare calibration metrics (ECE, AUROC) across all model runs on SWE-Rebench.

Usage:
    python scripts/compare_calibration.py [--outputs-dir outputs]

Generates a table like:
    Model           | ECE ↓   | AUROC ↑ | Brier ↓ | Pass@1
    ----------------|---------|---------|---------|-------
    DeepSeek V3.2   | 0.101   | 0.507   | 0.104   | 89.0%
    Claude Opus 4.5 | ...     | ...     | ...     | ...
    GPT-5-nano      | ...     | ...     | ...     | ...
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelResults:
    """Results for a single model run."""
    model: str
    task: str
    timestamp: str
    
    # Calibration metrics
    ece: float
    auroc: float
    brier_score: float
    spearman_rho: float
    pearson_rho: float
    mce: float
    
    # Performance metrics
    total_runs: int
    successful_runs: int
    success_rate: float
    mean_confidence: float
    
    # Outcome analysis
    mean_conf_success: float
    mean_conf_failure: float
    confidence_gap: float
    overconfidence_rate: float
    
    # Latency metrics
    mean_total_time: float = 0.0
    mean_step_latency: float = 0.0
    p95_total_time: float = 0.0
    p99_total_time: float = 0.0


def load_uncertainty_analysis(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load uncertainty analysis JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None


def extract_model_results(data: Dict[str, Any], timestamp: str) -> Optional[ModelResults]:
    """Extract model results from uncertainty analysis data."""
    try:
        # Get model name from first run
        if not data.get("runs"):
            return None
        
        first_run = data["runs"][0]
        model = first_run.get("agent", "unknown")
        task = first_run.get("task", "unknown")
        
        summary = data.get("summary", {})
        calibration = data.get("calibration", {})
        outcome = data.get("outcome_analysis", {})
        
        latency = data.get("latency", {})
        
        return ModelResults(
            model=model,
            task=task,
            timestamp=timestamp,
            ece=calibration.get("ece", 0.0),
            auroc=calibration.get("auroc", 0.5),
            brier_score=calibration.get("brier_score", 0.0),
            spearman_rho=calibration.get("spearman_rho", 0.0),
            pearson_rho=calibration.get("pearson_rho", 0.0),
            mce=calibration.get("mce", 0.0),
            total_runs=summary.get("total_runs", 0),
            successful_runs=summary.get("successful_runs", 0),
            success_rate=summary.get("success_rate", 0.0),
            mean_confidence=summary.get("mean_confidence", 0.0),
            mean_conf_success=outcome.get("mean_confidence_success", 0.0),
            mean_conf_failure=outcome.get("mean_confidence_failure", 0.0),
            confidence_gap=outcome.get("confidence_gap", 0.0),
            overconfidence_rate=outcome.get("overconfidence_rate", 0.0),
            # Latency metrics
            mean_total_time=latency.get("mean_total_time", 0.0),
            mean_step_latency=latency.get("mean_step_latency", 0.0),
            p95_total_time=latency.get("p95_total_time", 0.0),
            p99_total_time=latency.get("p99_total_time", 0.0),
        )
    except Exception as e:
        print(f"Warning: Could not extract results: {e}")
        return None


def find_swebench_runs(outputs_dir: Path) -> List[ModelResults]:
    """Find all SWE-bench runs with uncertainty analysis."""
    results = []
    
    for run_dir in sorted(outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        analysis_file = run_dir / "uncertainty_analysis.json"
        if not analysis_file.exists():
            continue
        
        data = load_uncertainty_analysis(analysis_file)
        if not data:
            continue
        
        # Check if this is a SWE-bench run
        runs = data.get("runs", [])
        if not runs:
            continue
        
        task = runs[0].get("task", "")
        if "swebench" not in task.lower():
            continue
        
        result = extract_model_results(data, run_dir.name)
        if result:
            results.append(result)
    
    return results


def print_comparison_table(results: List[ModelResults]) -> None:
    """Print a formatted comparison table."""
    if not results:
        print("No SWE-bench runs found with uncertainty analysis.")
        return
    
    # Group by model (take the latest run for each model)
    model_results: Dict[str, ModelResults] = {}
    for r in results:
        if r.model not in model_results or r.timestamp > model_results[r.model].timestamp:
            model_results[r.model] = r
    
    # Sort by AUROC descending
    sorted_results = sorted(model_results.values(), key=lambda x: x.auroc, reverse=True)
    
    # Print header
    print("\n" + "=" * 100)
    print("CALIBRATION QUALITY OF SAUP ACROSS ALL MODELS ON SWE-REBENCH")
    print("=" * 100)
    
    # Main metrics table
    print("\n### Main Calibration Metrics\n")
    header = f"{'Model':<20} | {'ECE ↓':>8} | {'AUROC ↑':>8} | {'Brier ↓':>8} | {'Pearson':>8} | {'Spearman':>8} | {'Pass@1':>8}"
    print(header)
    print("-" * len(header))
    
    for r in sorted_results:
        row = f"{r.model:<20} | {r.ece:>8.4f} | {r.auroc:>8.4f} | {r.brier_score:>8.4f} | {r.pearson_rho:>8.4f} | {r.spearman_rho:>8.4f} | {r.success_rate*100:>7.1f}%"
        print(row)
    
    # Confidence analysis table
    print("\n### Confidence Analysis\n")
    header2 = f"{'Model':<20} | {'Mean Conf':>10} | {'Conf|Succ':>10} | {'Conf|Fail':>10} | {'Gap':>8} | {'Overconf':>10}"
    print(header2)
    print("-" * len(header2))
    
    for r in sorted_results:
        row = f"{r.model:<20} | {r.mean_confidence:>10.4f} | {r.mean_conf_success:>10.4f} | {r.mean_conf_failure:>10.4f} | {r.confidence_gap:>8.4f} | {r.overconfidence_rate*100:>9.1f}%"
        print(row)
    
    # Latency analysis table
    print("\n### Latency Metrics\n")
    header3 = f"{'Model':<20} | {'Mean Time':>10} | {'Step Lat':>10} | {'P95 Time':>10} | {'P99 Time':>10}"
    print(header3)
    print("-" * len(header3))
    
    for r in sorted_results:
        row = f"{r.model:<20} | {r.mean_total_time:>9.1f}s | {r.mean_step_latency:>9.2f}s | {r.p95_total_time:>9.1f}s | {r.p99_total_time:>9.1f}s"
        print(row)
    
    # Interpretation
    print("\n### Interpretation\n")
    for r in sorted_results:
        print(f"**{r.model}** (n={r.total_runs}):")
        
        # ECE interpretation
        if r.ece < 0.05:
            ece_interp = "Well-calibrated"
        elif r.ece < 0.15:
            ece_interp = "Moderately calibrated"
        else:
            ece_interp = "Poorly calibrated"
        
        # AUROC interpretation
        if r.auroc > 0.7:
            auroc_interp = "Good discrimination"
        elif r.auroc > 0.6:
            auroc_interp = "Fair discrimination"
        else:
            auroc_interp = "Poor discrimination (near random)"
        
        # Spearman interpretation
        if abs(r.spearman_rho) > 0.5:
            spearman_interp = "Strong correlation"
        elif abs(r.spearman_rho) > 0.3:
            spearman_interp = "Moderate correlation"
        else:
            spearman_interp = "Weak correlation"
        
        print(f"  - ECE={r.ece:.3f}: {ece_interp}")
        print(f"  - AUROC={r.auroc:.3f}: {auroc_interp}")
        print(f"  - Spearman={r.spearman_rho:.3f}: {spearman_interp}")
        print(f"  - Confidence gap={r.confidence_gap:.4f}: {'Positive (good)' if r.confidence_gap > 0.05 else 'Flat (poor calibration)'}")
        print()
    
    print("=" * 100)


def print_latex_table(results: List[ModelResults]) -> None:
    """Print LaTeX formatted table for paper."""
    if not results:
        return
    
    # Group by model (take the latest run for each model)
    model_results: Dict[str, ModelResults] = {}
    for r in results:
        if r.model not in model_results or r.timestamp > model_results[r.model].timestamp:
            model_results[r.model] = r
    
    sorted_results = sorted(model_results.values(), key=lambda x: x.auroc, reverse=True)
    
    print("\n### LaTeX Table\n")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Calibration quality of SAUP across all models on SWE-Rebench}")
    print(r"\label{tab:calibration}")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"Model & Pearson $\uparrow$ & ECE $\downarrow$ & Brier $\downarrow$ & AUROC $\uparrow$ & Pass@1 \\")
    print(r"\midrule")
    
    for r in sorted_results:
        # Bold best values
        print(f"{r.model} & {r.pearson_rho:.3f} & {r.ece:.3f} & {r.brier_score:.3f} & {r.auroc:.3f} & {r.success_rate*100:.1f}\\% \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def export_csv(results: List[ModelResults], output_path: Path) -> None:
    """Export results to CSV."""
    import csv
    
    # Group by model
    model_results: Dict[str, ModelResults] = {}
    for r in results:
        if r.model not in model_results or r.timestamp > model_results[r.model].timestamp:
            model_results[r.model] = r
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Model', 'Pearson', 'ECE', 'AUROC', 'Brier', 'Spearman', 'MCE',
            'Pass@1', 'Total Runs', 'Mean Confidence',
            'Conf|Success', 'Conf|Failure', 'Confidence Gap', 'Overconfidence Rate',
            'Mean Time (s)', 'Mean Step Latency (s)', 'P95 Time (s)', 'P99 Time (s)'
        ])
        
        for r in model_results.values():
            writer.writerow([
                r.model, r.pearson_rho, r.ece, r.auroc, r.brier_score, r.spearman_rho, r.mce,
                r.success_rate, r.total_runs, r.mean_confidence,
                r.mean_conf_success, r.mean_conf_failure, r.confidence_gap, r.overconfidence_rate,
                r.mean_total_time, r.mean_step_latency, r.p95_total_time, r.p99_total_time
            ])
    
    print(f"\nCSV exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare calibration metrics across models")
    parser.add_argument("--outputs-dir", "-o", default="outputs", help="Directory containing run outputs")
    parser.add_argument("--latex", action="store_true", help="Also print LaTeX table")
    parser.add_argument("--csv", type=str, help="Export to CSV file")
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        print(f"Error: Outputs directory not found: {outputs_dir}")
        return
    
    print(f"Scanning {outputs_dir} for SWE-bench runs...")
    results = find_swebench_runs(outputs_dir)
    print(f"Found {len(results)} runs with uncertainty analysis")
    
    print_comparison_table(results)
    
    if args.latex:
        print_latex_table(results)
    
    if args.csv:
        export_csv(results, Path(args.csv))


if __name__ == "__main__":
    main()
