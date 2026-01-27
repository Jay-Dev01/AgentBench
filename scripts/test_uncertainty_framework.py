#!/usr/bin/env python3
"""
Test script for the uncertainty estimation framework.

This script demonstrates all components of the uncertainty framework:
1. Confidence extraction from API responses
2. Hierarchical uncertainty propagation
3. Calibration metrics (ECE, Brier, Temperature Scaling)
4. Scoring system
5. Error taxonomy
6. Agent wrapper for real-time tracking
7. Pipeline integration

Usage:
    python scripts/test_uncertainty_framework.py
"""

import sys
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
repo_root = script_dir.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import json
import random


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_confidence_extractor():
    """Test confidence extraction from various API response formats."""
    print_header("Testing ConfidenceExtractor")
    
    from agentbench_debug.uncertainty import ConfidenceExtractor
    
    extractor = ConfidenceExtractor(default_confidence=0.70)
    
    # Test 1: finish_reason = stop
    response_stop = {
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": "Hello world"}
        }]
    }
    conf, source = extractor.extract(response_stop)
    print(f"  finish_reason=stop: confidence={conf:.3f}, source={source}")
    assert conf == 0.85, f"Expected 0.85, got {conf}"
    
    # Test 2: finish_reason = tool_calls
    response_tool = {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {"tool_calls": [{"function": {"name": "test"}}]}
        }]
    }
    conf, source = extractor.extract(response_tool)
    print(f"  finish_reason=tool_calls: confidence={conf:.3f}, source={source}")
    assert conf == 0.80, f"Expected 0.80, got {conf}"
    
    # Test 3: Semantic hedging
    conf, source = extractor._extract_from_semantics("I think this might be correct, probably.")
    print(f"  Hedging text: confidence={conf:.3f}, source={source}")
    assert conf < 0.85, "Hedging should reduce confidence"
    
    # Test 4: No hedging
    conf, source = extractor._extract_from_semantics("The answer is 42.")
    print(f"  Clear text: confidence={conf:.3f}, source={source}")
    
    # Test 5: Self-reported confidence
    conf, source = extractor.extract_self_reported("I am 95% confident this is correct.")
    print(f"  Self-reported: confidence={conf}, source={source}")
    assert conf == 0.95, f"Expected 0.95, got {conf}"
    
    print("\n  [PASS] ConfidenceExtractor tests passed")


def test_hierarchical_uncertainty():
    """Test hierarchical uncertainty propagation."""
    print_header("Testing HierarchicalUncertainty")
    
    from agentbench_debug.uncertainty import HierarchicalUncertainty
    
    hierarchy = HierarchicalUncertainty(uncertainty_threshold=0.5)
    
    # Add some steps
    hierarchy.add_step(
        step_idx=0,
        action="look around",
        action_type="environment_action",
        observation="You see a kitchen.",
        confidence=0.85,
    )
    
    hierarchy.add_step(
        step_idx=1,
        action="open fridge",
        action_type="environment_action",
        observation="The fridge is empty.",
        confidence=0.75,
    )
    
    hierarchy.add_step(
        step_idx=2,
        action="go to living room",
        action_type="environment_action",
        observation="Nothing found here.",
        confidence=0.60,
        is_error=False,
    )
    
    # Compute trajectory metrics
    metrics = hierarchy.compute_trajectory_uncertainty()
    
    print(f"  Steps recorded: {metrics['n_steps']}")
    print(f"  Mean confidence: {metrics['mean_confidence']:.3f}")
    print(f"  Mean uncertainty: {metrics['mean_uncertainty']:.3f}")
    print(f"  Trend: {metrics['trend']}")
    print(f"  High uncertainty steps: {metrics['high_uncertainty_steps']}")
    
    # Get critical steps
    critical = hierarchy.get_critical_steps()
    print(f"  Critical steps (high uncertainty): {len(critical)}")
    
    print("\n  [PASS] HierarchicalUncertainty tests passed")


def test_calibration_metrics():
    """Test calibration metrics computation."""
    print_header("Testing CalibrationMetrics")
    
    from agentbench_debug.uncertainty import CalibrationMetrics, TemperatureScaling
    
    calibration = CalibrationMetrics(n_bins=10)
    
    # Generate synthetic data
    random.seed(42)
    confidences = []
    outcomes = []
    
    for _ in range(100):
        # Well-calibrated: high confidence -> more likely success
        conf = random.random()
        outcome = random.random() < conf
        confidences.append(conf)
        outcomes.append(outcome)
    
    result = calibration.compute(confidences, outcomes)
    
    print(f"  Samples: {result.n_samples}")
    print(f"  ECE (Expected Calibration Error): {result.ece:.4f}")
    print(f"  MCE (Maximum Calibration Error): {result.mce:.4f}")
    print(f"  Brier Score: {result.brier_score:.4f}")
    print(f"  Bins with data: {sum(1 for c in result.bin_counts if c > 0)}")
    
    # Test temperature scaling
    print("\n  Testing Temperature Scaling...")
    temp_scaler = TemperatureScaling()
    optimal_temp = temp_scaler.fit(confidences, outcomes, max_iters=50)
    print(f"  Optimal temperature: {optimal_temp:.3f}")
    
    # Test calibration adjustment
    test_conf = 0.8
    calibrated = temp_scaler.calibrate(test_conf)
    print(f"  Original: {test_conf:.3f} -> Calibrated: {calibrated:.3f}")
    
    print("\n  [PASS] CalibrationMetrics tests passed")


def test_scoring_system():
    """Test the scoring system."""
    print_header("Testing Scoring System")
    
    from agentbench_debug.uncertainty import Scorer, StepResult
    
    scorer = Scorer(progress_weight=0.3, completion_weight=0.7)
    
    # Create sample steps
    steps = [
        StepResult(step_idx=0, action="look", success=True, confidence=0.9),
        StepResult(step_idx=1, action="pick up", success=True, confidence=0.85),
        StepResult(step_idx=2, action="go to", success=False, confidence=0.6, error="Invalid action"),
        StepResult(step_idx=3, action="put", success=True, confidence=0.75),
    ]
    
    # Evaluate with task completion
    result = scorer.evaluate(steps, task_completed=True)
    
    print(f"  Progress score: {result.progress.score:.3f} ({result.progress.completed_steps}/{result.progress.total_steps})")
    print(f"  Completion score: {result.completion.score:.3f}")
    print(f"  Composite score: {result.composite.score:.3f}")
    print(f"  Mean confidence: {result.mean_confidence:.3f}")
    print(f"  Min confidence: {result.min_confidence:.3f}")
    
    # Test without completion
    result2 = scorer.evaluate(steps, task_completed=False)
    print(f"\n  Without completion:")
    print(f"    Composite score: {result2.composite.score:.3f}")
    
    print("\n  [PASS] Scoring System tests passed")


def test_error_taxonomy():
    """Test error classification."""
    print_header("Testing ErrorTaxonomy")
    
    from agentbench_debug.uncertainty import ErrorTaxonomy, ErrorCategory
    
    taxonomy = ErrorTaxonomy()
    
    # Test various error messages
    test_errors = [
        ("Rate limit exceeded. Please retry.", ErrorCategory.RATE_LIMIT),
        ("401 Unauthorized: Invalid API key", ErrorCategory.AUTHENTICATION),
        ("Request timeout after 30 seconds", ErrorCategory.TIMEOUT),
        ("Context length exceeds maximum", ErrorCategory.CONTEXT_LIMIT),
        ("Connection refused", ErrorCategory.NETWORK),
    ]
    
    for error_msg, expected_category in test_errors:
        record = taxonomy.record_error(error_msg, step_idx=0)
        print(f"  '{error_msg[:40]}...'")
        print(f"    Category: {record.category.value}, Severity: {record.severity.value}")
        print(f"    Recovery: {record.recovery_action}")
        assert record.category == expected_category, f"Expected {expected_category}, got {record.category}"
    
    # Get summary
    summary = taxonomy.get_summary()
    print(f"\n  Error Summary:")
    print(f"    Total errors: {summary['total_errors']}")
    print(f"    By category: {summary['by_category']}")
    
    print("\n  [PASS] ErrorTaxonomy tests passed")


def test_orchestration_harness():
    """Test the unified orchestration harness."""
    print_header("Testing OrchestrationHarness")
    
    from agentbench_debug.uncertainty import OrchestrationHarness
    
    harness = OrchestrationHarness(
        task_id="test-task-001",
        task_type="alfworld",
    )
    
    # Simulate a workflow
    harness.start_workflow(task_id="test-001", task_type="alfworld")
    
    # Record steps with mock API responses
    mock_responses = [
        {"choices": [{"finish_reason": "tool_calls", "message": {}}]},
        {"choices": [{"finish_reason": "tool_calls", "message": {}}]},
        {"choices": [{"finish_reason": "stop", "message": {}}]},
    ]
    
    for i, resp in enumerate(mock_responses):
        harness.record_step(
            action=f"action_{i}",
            action_type="environment_action",
            observation=f"observation_{i}",
            success=True,
            raw_api_response=resp,
        )
    
    # Finish and get results
    result = harness.finish_workflow(task_completed=True)
    
    print(f"  Task: {result.task_id}")
    print(f"  Success: {result.success}")
    print(f"  Steps: {len(result.steps)}")
    print(f"  Mean confidence: {result.evaluation.mean_confidence:.3f}")
    print(f"  Composite score: {result.evaluation.composite.score:.3f}")
    print(f"  Trajectory trend: {result.trajectory_metrics.get('trend', 'unknown')}")
    
    print("\n  [PASS] OrchestrationHarness tests passed")


def test_agent_wrapper():
    """Test the agent wrapper functionality."""
    print_header("Testing Agent Wrapper")
    
    from agentbench_debug.uncertainty import UncertaintyTracker
    
    # Create a tracker (wrapper would use this internally)
    tracker = UncertaintyTracker(task_id="test", task_type="dbbench")
    
    # Simulate inference calls
    mock_responses = [
        {"choices": [{"finish_reason": "tool_calls"}]},
        {"choices": [{"finish_reason": "tool_calls"}]},
        {"choices": [{"finish_reason": "stop"}]},
    ]
    
    for i, resp in enumerate(mock_responses):
        step = tracker.record_inference(
            input_messages=[{"role": "user", "content": f"Query {i}"}],
            output=f"SELECT * FROM table_{i}",
            raw_response=resp,
            action_type="query",
        )
        print(f"  Step {i}: confidence={step.confidence:.3f} ({step.confidence_source})")
    
    # Get summary
    summary = tracker.get_summary()
    print(f"\n  Summary:")
    print(f"    Steps: {summary['n_steps']}")
    print(f"    Mean confidence: {summary['mean_confidence']:.3f}")
    print(f"    Trend: {summary['trend']}")
    
    print("\n  [PASS] Agent Wrapper tests passed")


def test_pipeline_integration():
    """Test pipeline integration helpers."""
    print_header("Testing Pipeline Integration")
    
    from agentbench_debug.uncertainty import (
        UncertaintyReport,
        UncertaintyCallback,
    )
    
    # Test UncertaintyReport
    runs = [
        {"task_type": "alfworld", "success": True, "mean_confidence": 0.85},
        {"task_type": "alfworld", "success": False, "mean_confidence": 0.60},
        {"task_type": "dbbench", "success": True, "mean_confidence": 0.90},
        {"task_type": "dbbench", "success": True, "mean_confidence": 0.88},
    ]
    
    report = UncertaintyReport.from_runs(runs)
    
    print(f"  Report Summary:")
    print(f"    Total runs: {report.total_runs}")
    print(f"    Successful: {report.successful_runs}")
    print(f"    Mean confidence: {report.mean_confidence:.3f}")
    print(f"    By task type: {report.confidence_by_task_type}")
    
    # Test callback
    callback = UncertaintyCallback(print_steps=False)
    tracker = callback.on_run_start("test-task", "alfworld")
    callback.on_run_end(success=True)
    
    final_report = callback.get_report()
    print(f"    Callback runs tracked: {final_report.total_runs}")
    
    print("\n  [PASS] Pipeline Integration tests passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  UNCERTAINTY FRAMEWORK TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Confidence Extractor", test_confidence_extractor),
        ("Hierarchical Uncertainty", test_hierarchical_uncertainty),
        ("Calibration Metrics", test_calibration_metrics),
        ("Scoring System", test_scoring_system),
        ("Error Taxonomy", test_error_taxonomy),
        ("Orchestration Harness", test_orchestration_harness),
        ("Agent Wrapper", test_agent_wrapper),
        ("Pipeline Integration", test_pipeline_integration),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"  TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

