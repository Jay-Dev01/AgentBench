#!/usr/bin/env python3
"""
Recalculate Pearson correlation from existing uncertainty analysis files.

Usage:
    python scripts/recalculate_pearson.py
"""

import json
import math
from pathlib import Path
from typing import List


def compute_pearson(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    
    denominator = math.sqrt(var_x * var_y)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def process_file(filepath: Path) -> None:
    """Process a single uncertainty analysis file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    runs = data.get("runs", [])
    if not runs:
        return
    
    # Extract confidence and outcomes
    confidences = [r.get("mean_confidence", 0.5) for r in runs]
    outcomes = [1.0 if r.get("success", False) else 0.0 for r in runs]
    
    # Calculate Pearson
    pearson = compute_pearson(confidences, outcomes)
    
    # Update calibration section
    if "calibration" not in data:
        data["calibration"] = {}
    
    data["calibration"]["pearson_rho"] = pearson
    
    # Save back
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    model = runs[0].get("agent", "unknown")
    print(f"{model}: Pearson = {pearson:.4f}")


def main():
    outputs_dir = Path("outputs")
    
    print("Recalculating Pearson correlation for existing runs...\n")
    
    for run_dir in sorted(outputs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        analysis_file = run_dir / "uncertainty_analysis.json"
        if not analysis_file.exists():
            continue
        
        # Check if SWE-bench run
        with open(analysis_file, 'r') as f:
            data = json.load(f)
        
        runs = data.get("runs", [])
        if not runs:
            continue
        
        task = runs[0].get("task", "")
        if "swebench" not in task.lower():
            continue
        
        print(f"Processing {run_dir.name}...")
        process_file(analysis_file)
    
    print("\nDone! Run compare_calibration.py to see updated results:")
    print("  python scripts/compare_calibration.py --csv calibration_results.csv")


if __name__ == "__main__":
    main()
