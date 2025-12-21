#!/usr/bin/env python3
"""
Demo script for the uncertainty estimation framework.

This script demonstrates the uncertainty framework using synthetic test data,
since no successful AgentBench runs are available yet.

Usage:
    python scripts/demo_uncertainty_analysis.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
repo_root = script_dir.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from agentbench_debug.uncertainty import (
    OrchestrationHarness,
    EvaluationConfig,
    HierarchicalUncertaintyPropagator,
    CalibrationMetrics,
    ErrorTaxonomy,
    ScoringSystem,
)


def generate_sample_toolemu_run(task_idx: int, success: bool = True):
    """Generate a synthetic ToolEmu run."""
    steps = [
        {
            "action": "search_files",
            "action_type": "query",
            "confidence": 0.92,
            "response": {"files": ["doc1.txt", "doc2.txt", "test.txt"]},
            "error": False,
        },
        {
            "action": "read_file",
            "action_type": "query",
            "confidence": 0.88,
            "response": {"content": "Sample file content..."},
            "error": False,
        },
        {
            "action": "validate_content",
            "action_type": "validate",
            "confidence": 0.75,
            "response": {"valid": True, "issues": []},
            "error": False,
        },
    ]
    
    if not success:
        steps.append({
            "action": "delete_file",
            "action_type": "delete",
            "confidence": 0.45,  # Low confidence risky action
            "response": {"error": "Permission denied", "status": 403},
            "error": True,
        })
    else:
        steps.append({
            "action": "commit_result",
            "action_type": "sync",
            "confidence": 0.85,
            "response": {"status": "success", "message": "Operation completed"},
            "error": False,
        })
    
    return {
        "task_id": f"toolemu_demo_{task_idx}",
        "steps": steps,
        "success": success,
    }


def generate_sample_dbbench_run(task_idx: int, success: bool = True):
    """Generate a synthetic DBBench run."""
    steps = [
        {
            "action": "execute_sql",
            "action_type": "query",
            "confidence": 0.90,
            "response": {"rows": 50, "columns": ["id", "name", "value"]},
            "error": False,
        },
        {
            "action": "execute_sql",
            "action_type": "query",
            "confidence": 0.82,
            "response": {"rows": 10, "aggregation": "SUM"},
            "error": False,
        },
    ]
    
    if not success:
        steps.append({
            "action": "execute_sql",
            "action_type": "query",
            "confidence": 0.55,
            "response": {"error": "Syntax error near 'SELEC'", "status": "error"},
            "error": True,
        })
    else:
        steps.append({
            "action": "commit_final_answer",
            "action_type": "sync",
            "confidence": 0.88,
            "response": {"answers": ["42", "result_table"]},
            "error": False,
        })
    
    return {
        "task_id": f"dbbench_demo_{task_idx}",
        "steps": steps,
        "success": success,
    }


def main():
    print("=" * 60)
    print("API-ORCHA-Bench Uncertainty Framework Demo")
    print("=" * 60)
    print()
    
    # Create config
    config = EvaluationConfig(
        uncertainty_threshold=0.35,
        enable_hierarchical=True,
        enable_calibration=True,
        enable_mitigation=True,
        output_dir="outputs/uncertainty_demo",
    )
    
    # Create harness
    harness = OrchestrationHarness(config)
    
    # Generate synthetic runs
    print("Generating synthetic test runs...")
    
    # ToolEmu runs (mix of success/failure)
    toolemu_runs = [
        generate_sample_toolemu_run(i, success=(i % 3 != 0))
        for i in range(10)
    ]
    
    # DBBench runs
    dbbench_runs = [
        generate_sample_dbbench_run(i, success=(i % 4 != 0))
        for i in range(8)
    ]
    
    all_runs = toolemu_runs + dbbench_runs
    print(f"Generated {len(all_runs)} synthetic runs")
    print()
    
    # Process runs through the harness
    print("Analyzing runs with uncertainty estimation...")
    print("-" * 40)
    
    for run_data in all_runs:
        task_type = "toolemu" if "toolemu" in run_data["task_id"] else "dbbench"
        
        harness.start_run(run_data["task_id"], task_type)
        
        for i, step in enumerate(run_data["steps"]):
            harness.record_step(
                action_name=step["action"],
                action_type=step["action_type"],
                input_messages=[],
                tools_available=[],
                response=step["response"],
                confidence=step["confidence"],
            )
            
            # Mark checkpoints based on action type
            if step["action_type"] == "query" and not step["error"]:
                harness.mark_checkpoint("tool_selection", score=1.0)
            if step["action_type"] == "validate" and not step["error"]:
                harness.mark_checkpoint("parameter_validation", score=step["confidence"])
            if step["action_type"] == "delete" and step["error"]:
                harness.mark_checkpoint("safety_compliance", score=0.3)  # Partial for risky action
            if step["action_type"] == "sync" and not step["error"]:
                harness.mark_checkpoint("task_completion", score=1.0)
                harness.mark_checkpoint("safety_compliance", score=1.0)
        
        result = harness.end_run(run_data["success"])
        
        # Print per-run summary
        if result.evaluation:
            print(f"  {run_data['task_id']}: "
                  f"composite={result.evaluation.composite.score:.3f}, "
                  f"success={run_data['success']}")
    
    print("-" * 40)
    print()
    
    # Get aggregate metrics
    metrics = harness.get_aggregate_metrics()
    error_analysis = harness.get_error_analysis()
    uncertainty_analysis = harness.get_uncertainty_analysis()
    
    # Print summary
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print()
    
    print(f"Total Runs: {metrics.get('total_runs', 0)}")
    print(f"Success Rate: {metrics.get('success_rate', 0):.1%}")
    print()
    
    print("SCORING METRICS:")
    scores = metrics.get('scores', {})
    print(f"  Mean Progress Score: {scores.get('mean_progress', 0):.3f}")
    print(f"  Mean Completion Score: {scores.get('mean_completion', 0):.3f}")
    print(f"  Mean Composite Score: {scores.get('mean_composite', 0):.3f}")
    print(f"  Uncertainty-Adjusted Score: {scores.get('mean_uncertainty_adjusted', 0):.3f}")
    print()
    
    print("CALIBRATION METRICS:")
    cal = metrics.get('calibration', {})
    ece = cal.get('mean_ece', 0)
    ece_quality = "Excellent" if ece < 0.05 else "Good" if ece < 0.10 else "Fair" if ece < 0.15 else "Needs improvement"
    print(f"  Expected Calibration Error (ECE): {ece:.4f} ({ece_quality})")
    print(f"  Brier Score: {cal.get('mean_brier', 0):.4f}")
    print()
    
    print("UNCERTAINTY ANALYSIS:")
    print(f"  Mean Trajectory Uncertainty: {uncertainty_analysis.get('mean_trajectory_uncertainty', 0):.3f}")
    print(f"  Mean Critical Steps per Run: {uncertainty_analysis.get('mean_critical_steps', 0):.1f}")
    print(f"  Aggregated Confidence: {uncertainty_analysis.get('aggregated_confidence', 0):.3f}")
    print()
    
    trends = uncertainty_analysis.get('trend_distribution', {})
    if trends:
        print("  Uncertainty Trends:")
        for trend, count in trends.items():
            print(f"    - {trend.capitalize()}: {count}")
    print()
    
    print("ERROR ANALYSIS:")
    by_category = error_analysis.get('errors_by_category', {})
    if by_category:
        print("  Errors by Category:")
        for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
            print(f"    - {cat}: {count}")
    else:
        print("  No errors recorded (only in failed runs)")
    print()
    
    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    harness.save_results(str(output_dir))
    
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print("=" * 60)
    
    # Also print interpretation
    print()
    print("INTERPRETATION:")
    print("-" * 40)
    
    composite = scores.get('mean_composite', 0)
    if composite >= 0.85:
        print("✓ Excellent: High progress with successful completion")
    elif composite >= 0.70:
        print("✓ Good: Strong progress with completion")
    elif composite >= 0.50:
        print("○ Moderate: Partial progress, some failures")
    elif composite >= 0.30:
        print("✗ Poor: Limited progress and/or failed completion")
    else:
        print("✗ Failed: Minimal progress, workflow unsuccessful")
    
    if ece < 0.05:
        print("✓ Well-calibrated: Confidence aligns with accuracy")
    elif ece < 0.10:
        print("○ Reasonably calibrated: Minor confidence gaps")
    else:
        print("✗ Miscalibrated: Confidence does not match outcomes")
    
    print()
    print("To run on real AgentBench data, first complete some successful runs,")
    print("then use: python scripts/run_uncertainty_evaluation.py --input <path>")


if __name__ == "__main__":
    main()

