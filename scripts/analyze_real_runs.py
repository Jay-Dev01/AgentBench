#!/usr/bin/env python3
"""
Analyze Real AgentBench FC Runs with Uncertainty Estimation.

This script analyzes actual run data from AgentBench FC outputs using
the uncertainty estimation framework.

Supported Tasks:
    - alfworld-std (AF): Household tasks
    - dbbench-std (DB): Database SQL tasks  
    - os-std (OS): OS interaction tasks
    - kg-std (KG): Knowledge graph QA
    - webshop-std (WS): Web shopping

Usage:
    # Analyze a specific runs file
    python scripts/analyze_real_runs.py --input outputs/2025-12-08-16-44-37/gpt-4o-mini/alfworld-std/runs.jsonl
    
    # Analyze all runs in outputs directory
    python scripts/analyze_real_runs.py --all
    
    # Analyze with verbose output
    python scripts/analyze_real_runs.py --input path/to/runs.jsonl --verbose
"""

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

from agentbench_debug.uncertainty import (
    OrchestrationHarness,
    EvaluationConfig,
    ConfidenceExtractor,
    UncertaintyTracker,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze AgentBench runs with uncertainty estimation")
    parser.add_argument("--input", "-i", type=str, help="Path to runs.jsonl file")
    parser.add_argument("--all", action="store_true", help="Analyze all runs in outputs directory")
    parser.add_argument("--output", "-o", type=str, default="outputs/uncertainty_analysis", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of runs to analyze")
    return parser.parse_args()


def detect_task_type(path: Path) -> str:
    """Detect task type from path for AgentBench FC tasks."""
    path_str = str(path).lower()
    
    # AgentBench FC Core Tasks (priority order)
    if "alfworld" in path_str:
        return "alfworld"
    if "dbbench" in path_str:
        return "dbbench"
    if "os-std" in path_str or "os_interaction" in path_str:
        return "os"
    if "knowledgegraph" in path_str or "kg-std" in path_str or "/kg/" in path_str:
        return "kg"
    if "webshop" in path_str:
        return "webshop"
    
    return "unknown"


def extract_confidence_from_content(content: str, extractor: ConfidenceExtractor) -> float:
    """Extract confidence from agent response content."""
    if not content:
        return 0.7  # Default
    
    signals = extractor.extract(content, api_type="generic")
    return signals.confidence


def analyze_run(
    run: Dict[str, Any],
    task_type: str,
    extractor: ConfidenceExtractor,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Analyze a single run for uncertainty."""
    tracker = UncertaintyTracker()
    
    output = run.get("output", {})
    history = output.get("history", [])
    success = output.get("result", False)
    index = run.get("index", 0)
    
    if not history:
        return None
    
    # Process each turn in the history
    agent_responses = []
    step_confidences = []
    
    for i, item in enumerate(history):
        role = item.get("role", "")
        content = item.get("content", "")
        
        if role == "agent" and content:
            # Extract confidence from agent response
            confidence = extract_confidence_from_content(content, extractor)
            
            # Infer action type from content for AgentBench FC tasks
            action_type = "respond"
            content_lower = content.lower()
            
            # ALFWorld - household actions
            if any(kw in content_lower for kw in ["take_action", "go to", "pick up", "put", "open", "close", "toggle"]):
                action_type = "environment_action"
            # DBBench - SQL queries
            elif any(kw in content_lower for kw in ["execute_sql", "select", "insert", "update"]):
                action_type = "query"
            elif "commit_final_answer" in content_lower:
                action_type = "submit"
            # OS Interaction - shell commands
            elif any(kw in content_lower for kw in ["bash_action", "cd ", "ls ", "cat ", "grep "]):
                action_type = "shell_command"
            elif any(kw in content_lower for kw in ["finish_action", "answer_action"]):
                action_type = "submit"
            # WebShop - e-commerce
            elif "search_action" in content_lower or "search for" in content_lower:
                action_type = "search"
            elif "click_action" in content_lower:
                action_type = "navigation"
            # Generic patterns
            elif "error" in content_lower or "unable" in content_lower:
                action_type = "error_handling"
                confidence *= 0.9  # Reduce confidence for errors
            elif "clarify" in content_lower or "confirm" in content_lower:
                action_type = "clarify"
            elif "search" in content_lower or "find" in content_lower:
                action_type = "query"
            elif "delete" in content_lower or "remove" in content_lower:
                action_type = "delete"
            elif "create" in content_lower or "write" in content_lower:
                action_type = "write"
            
            # Record in tracker
            tracker.record_response(
                content=content,
                action_name=f"step_{i}",
                action_type=action_type,
            )
            
            agent_responses.append({
                "step": i,
                "confidence": confidence,
                "action_type": action_type,
                "preview": content[:100] + "..." if len(content) > 100 else content,
            })
            step_confidences.append(confidence)
    
    if not step_confidences:
        return None
    
    # Get analysis
    analysis = tracker.get_analysis()
    
    result = {
        "index": index,
        "task_type": task_type,
        "success": success,
        "n_steps": len(step_confidences),
        "mean_confidence": analysis["mean_confidence"],
        "min_confidence": analysis["min_confidence"],
        "trajectory_uncertainty": analysis["trajectory_uncertainty"],
        "final_confidence": analysis["final_confidence"],
        "trend": analysis["trend"],
        "high_uncertainty_count": analysis["high_uncertainty_count"],
        "step_confidences": step_confidences,
    }
    
    if verbose:
        print(f"\n  Run {index}:")
        print(f"    Success: {success}")
        print(f"    Steps: {len(step_confidences)}")
        print(f"    Mean confidence: {analysis['mean_confidence']:.3f}")
        print(f"    Trend: {analysis['trend']}")
    
    return result


def analyze_file(
    file_path: Path,
    output_dir: Path,
    verbose: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Analyze all runs in a file."""
    task_type = detect_task_type(file_path)
    extractor = ConfidenceExtractor()
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {file_path}")
    print(f"Task type: {task_type}")
    print(f"{'='*60}")
    
    # Load runs
    runs = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    if limit:
        runs = runs[:limit]
    
    print(f"Loaded {len(runs)} runs")
    
    # Analyze each run
    results = []
    successful_runs = 0
    failed_runs = 0
    
    for run in runs:
        result = analyze_run(run, task_type, extractor, verbose)
        if result:
            results.append(result)
            if result["success"]:
                successful_runs += 1
            else:
                failed_runs += 1
    
    if not results:
        print("No valid runs to analyze")
        return None
    
    # Compute aggregate statistics
    all_confidences = [r["mean_confidence"] for r in results]
    all_uncertainties = [r["trajectory_uncertainty"] for r in results]
    trends = {"increasing": 0, "decreasing": 0, "stable": 0}
    for r in results:
        trends[r["trend"]] = trends.get(r["trend"], 0) + 1
    
    # Success vs failure analysis
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    summary = {
        "file": str(file_path),
        "task_type": task_type,
        "total_runs": len(results),
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
        "success_rate": successful_runs / len(results) if results else 0,
        "overall": {
            "mean_confidence": sum(all_confidences) / len(all_confidences),
            "min_confidence": min(all_confidences),
            "max_confidence": max(all_confidences),
            "mean_uncertainty": sum(all_uncertainties) / len(all_uncertainties),
        },
        "trend_distribution": trends,
        "by_outcome": {
            "successful": {
                "count": len(successful_results),
                "mean_confidence": sum(r["mean_confidence"] for r in successful_results) / len(successful_results) if successful_results else 0,
            },
            "failed": {
                "count": len(failed_results),
                "mean_confidence": sum(r["mean_confidence"] for r in failed_results) / len(failed_results) if failed_results else 0,
            },
        },
        "runs": results,
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total runs: {summary['total_runs']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"\nConfidence Metrics:")
    print(f"  Mean confidence: {summary['overall']['mean_confidence']:.3f}")
    print(f"  Min confidence: {summary['overall']['min_confidence']:.3f}")
    print(f"  Max confidence: {summary['overall']['max_confidence']:.3f}")
    print(f"  Mean uncertainty: {summary['overall']['mean_uncertainty']:.3f}")
    print(f"\nTrend Distribution:")
    for trend, count in trends.items():
        print(f"  {trend}: {count} ({count/len(results)*100:.1f}%)")
    print(f"\nBy Outcome:")
    print(f"  Successful runs ({summary['by_outcome']['successful']['count']}): mean_conf={summary['by_outcome']['successful']['mean_confidence']:.3f}")
    print(f"  Failed runs ({summary['by_outcome']['failed']['count']}): mean_conf={summary['by_outcome']['failed']['mean_confidence']:.3f}")
    
    # Correlation insight
    if successful_results and failed_results:
        conf_diff = summary['by_outcome']['successful']['mean_confidence'] - summary['by_outcome']['failed']['mean_confidence']
        if conf_diff > 0.05:
            print(f"\n[INSIGHT] Successful runs have higher confidence (+{conf_diff:.3f})")
        elif conf_diff < -0.05:
            print(f"\n[INSIGHT] Failed runs have higher confidence ({conf_diff:.3f}) - may indicate overconfidence")
        else:
            print(f"\n[INSIGHT] Confidence is similar between success/failure - calibration may be poor")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = file_path.parent.name.replace("-", "_")
    
    output_file = output_dir / f"uncertainty_{task_name}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return summary


def find_all_runs_files(outputs_dir: Path) -> List[Path]:
    """Find all runs.jsonl files in outputs directory."""
    return list(outputs_dir.rglob("runs.jsonl"))


def main():
    args = parse_args()
    output_dir = Path(args.output)
    
    if args.all:
        # Analyze all runs files
        outputs_dir = repo_root / "outputs"
        run_files = find_all_runs_files(outputs_dir)
        
        if not run_files:
            print("No runs.jsonl files found in outputs directory")
            return
        
        print(f"Found {len(run_files)} run files to analyze")
        
        all_summaries = []
        for run_file in run_files:
            summary = analyze_file(run_file, output_dir, args.verbose, args.limit)
            if summary:
                all_summaries.append(summary)
        
        # Print overall summary
        if all_summaries:
            print(f"\n{'='*60}")
            print("OVERALL SUMMARY ACROSS ALL FILES")
            print(f"{'='*60}")
            
            total_runs = sum(s["total_runs"] for s in all_summaries)
            total_success = sum(s["successful_runs"] for s in all_summaries)
            avg_confidence = sum(s["overall"]["mean_confidence"] * s["total_runs"] for s in all_summaries) / total_runs
            
            print(f"Files analyzed: {len(all_summaries)}")
            print(f"Total runs: {total_runs}")
            print(f"Overall success rate: {total_success/total_runs:.1%}")
            print(f"Average confidence: {avg_confidence:.3f}")
    
    elif args.input:
        # Analyze single file
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"File not found: {input_path}")
            return
        
        analyze_file(input_path, output_dir, args.verbose, args.limit)
    
    else:
        print("Please specify --input <file> or --all")
        print("\nAvailable run files:")
        outputs_dir = repo_root / "outputs"
        for f in find_all_runs_files(outputs_dir)[:10]:
            print(f"  {f.relative_to(repo_root)}")
        if len(list(find_all_runs_files(outputs_dir))) > 10:
            print("  ...")


if __name__ == "__main__":
    main()

