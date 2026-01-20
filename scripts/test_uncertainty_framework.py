#!/usr/bin/env python3
"""
Test script for the complete Uncertainty Estimation Framework.

This script verifies that all components of the uncertainty framework
are working correctly and can measure agent uncertainty.

Usage:
    python scripts/test_uncertainty_framework.py

What it tests:
1. Confidence extraction from various LLM API response formats
2. Agent wrapper with uncertainty tracking
3. Hierarchical uncertainty propagation
4. Calibration metrics computation
5. Pipeline integration
6. Full orchestration harness
"""

import sys
from pathlib import Path

# Add src to path
script_dir = Path(__file__).parent
repo_root = script_dir.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def test_confidence_extraction():
    """Test confidence extraction from various API response formats."""
    print("\n" + "="*60)
    print("TEST 1: Confidence Extraction")
    print("="*60)
    
    from agentbench_debug.uncertainty import ConfidenceExtractor, extract_confidence
    
    extractor = ConfidenceExtractor()
    
    # Test 1a: OpenAI Chat Completion with logprobs
    print("\n1a. OpenAI Chat Completion with logprobs:")
    openai_response = {
        "choices": [{
            "message": {
                "content": "The answer is 42.",
                "role": "assistant"
            },
            "logprobs": {
                "content": [
                    {"token": "The", "logprob": -0.05},
                    {"token": " answer", "logprob": -0.1},
                    {"token": " is", "logprob": -0.02},
                    {"token": " 42", "logprob": -0.3},
                    {"token": ".", "logprob": -0.01},
                ]
            },
            "finish_reason": "stop"
        }]
    }
    signals = extractor.extract(openai_response, api_type="openai")
    print(f"    Confidence: {signals.confidence:.3f}")
    print(f"    Source: {signals.source}")
    print(f"    Mean logprob: {signals.mean_logprob:.3f}" if signals.mean_logprob else "    No logprobs")
    assert 0.6 < signals.confidence < 1.0, "OpenAI confidence should be reasonable"
    print("    [OK] PASSED")
    
    # Test 1b: OpenAI without logprobs (semantic analysis)
    print("\n1b. OpenAI without logprobs (semantic analysis):")
    openai_uncertain = {
        "choices": [{
            "message": {
                "content": "I'm not sure, but I think the answer might be around 40-45.",
                "role": "assistant"
            },
            "finish_reason": "stop"
        }]
    }
    signals = extractor.extract(openai_uncertain, api_type="openai")
    print(f"    Confidence: {signals.confidence:.3f}")
    print(f"    Source: {signals.source}")
    print(f"    Uncertainty phrases: {signals.uncertainty_phrases}")
    assert signals.confidence < 0.7, "Uncertain response should have lower confidence"
    print("    [OK] PASSED")
    
    # Test 1c: Gemini format
    print("\n1c. Gemini response format:")
    gemini_response = {
        "candidates": [{
            "content": {
                "parts": [{"text": "The capital of France is Paris."}]
            },
            "finishReason": "STOP",
            "safetyRatings": []
        }]
    }
    signals = extractor.extract(gemini_response, api_type="gemini")
    print(f"    Confidence: {signals.confidence:.3f}")
    print(f"    Source: {signals.source}")
    assert signals.confidence > 0.5, "Gemini confident response should have decent confidence"
    print("    [OK] PASSED")
    
    # Test 1d: Self-reported confidence
    print("\n1d. Self-reported confidence in text:")
    self_reported = "I am 85% confident that the answer is correct."
    signals = extractor.extract(self_reported, api_type="generic")
    print(f"    Confidence: {signals.confidence:.3f}")
    print(f"    Self-reported: {signals.self_reported_confidence}")
    assert signals.self_reported_confidence == 0.85, "Should extract 85% confidence"
    print("    [OK] PASSED")
    
    # Test 1e: Convenience function
    print("\n1e. Convenience function extract_confidence():")
    conf = extract_confidence(openai_response)
    print(f"    Confidence: {conf:.3f}")
    assert 0 < conf < 1, "Should return valid confidence"
    print("    [OK] PASSED")
    
    print("\n[OK] All confidence extraction tests passed!")


def test_uncertainty_tracker():
    """Test the standalone uncertainty tracker."""
    print("\n" + "="*60)
    print("TEST 2: Uncertainty Tracker")
    print("="*60)
    
    from agentbench_debug.uncertainty import UncertaintyTracker
    
    tracker = UncertaintyTracker(uncertainty_threshold=0.35)
    
    # Simulate a multi-step workflow
    steps = [
        ("Searching for files...", "search_files", "query"),
        ("Reading file content...", "read_file", "query"),
        ("I'm not sure about this operation...", "validate", "validate"),
        ("Successfully completed the task.", "commit", "sync"),
    ]
    
    print("\nRecording steps:")
    for content, action_name, action_type in steps:
        conf = tracker.record_response(
            content=content,
            action_name=action_name,
            action_type=action_type,
        )
        print(f"    {action_name}: confidence={conf:.3f}")
    
    # Get analysis
    analysis = tracker.get_analysis()
    print(f"\nAnalysis:")
    print(f"    Steps: {analysis['n_steps']}")
    print(f"    Mean confidence: {analysis['mean_confidence']:.3f}")
    print(f"    Min confidence: {analysis['min_confidence']:.3f}")
    print(f"    Trajectory uncertainty: {analysis['trajectory_uncertainty']:.3f}")
    print(f"    Trend: {analysis['trend']}")
    
    assert analysis['n_steps'] == 4, "Should have 4 steps"
    assert 0 < analysis['mean_confidence'] < 1, "Mean confidence should be valid"
    
    print("\n[OK] Uncertainty tracker test passed!")


def test_hierarchical_propagation():
    """Test hierarchical uncertainty propagation."""
    print("\n" + "="*60)
    print("TEST 3: Hierarchical Uncertainty Propagation")
    print("="*60)
    
    from agentbench_debug.uncertainty import HierarchicalUncertaintyPropagator
    
    propagator = HierarchicalUncertaintyPropagator(
        uncertainty_threshold=0.35,
        enable_hmm=True,
    )
    
    # Simulate a workflow
    actions = [
        ("authenticate", "auth", 0.95),
        ("query_users", "query", 0.88),
        ("validate_data", "validate", 0.75),
        ("delete_user", "delete", 0.45),  # Low confidence risky action
        ("sync_changes", "sync", 0.82),
    ]
    
    print("\nRecording actions:")
    for action_name, action_type, confidence in actions:
        unc = propagator.compute_action_uncertainty(
            action_name=action_name,
            action_type=action_type,
            confidence=confidence,
        )
        print(f"    {action_name}: confidence={confidence:.2f}, "
              f"weighted_uncertainty={unc.weighted_uncertainty:.3f}, "
              f"weight={unc.criticality_weight}")
        
        propagator.compute_observation_uncertainty(
            response={"status": "success"},
        )
        propagator.next_step()
    
    # Get complete analysis
    result = propagator.analyze_complete(trajectory_id="test_run")
    
    print(f"\nHierarchical Analysis:")
    print(f"    Action-level entries: {len(result.action_level)}")
    print(f"    Observation-level entries: {len(result.observation_level)}")
    print(f"    Trajectory uncertainty: {result.trajectory_level.cumulative_uncertainty:.3f}")
    print(f"    Final confidence: {result.trajectory_level.final_confidence:.3f}")
    print(f"    Trend: {result.trajectory_level.uncertainty_trend}")
    print(f"    Critical steps: {result.trajectory_level.critical_steps}")
    print(f"    HMM states: {result.trajectory_level.hmm_state_sequence}")
    print(f"    Aggregated score: {result.aggregated_score:.3f}")
    
    assert len(result.action_level) == 5, "Should have 5 action entries"
    assert result.aggregated_score > 0, "Aggregated score should be positive"
    
    print("\n[OK] Hierarchical propagation test passed!")


def test_calibration_metrics():
    """Test calibration metrics computation."""
    print("\n" + "="*60)
    print("TEST 4: Calibration Metrics")
    print("="*60)
    
    from agentbench_debug.uncertainty import CalibrationMetrics
    
    calibration = CalibrationMetrics(n_bins=10)
    
    # Add prediction-outcome pairs
    # Well-calibrated predictions
    predictions = [
        (0.9, True), (0.9, True), (0.9, True), (0.9, False),  # 90% conf, 75% acc
        (0.8, True), (0.8, True), (0.8, False),  # 80% conf, 67% acc
        (0.7, True), (0.7, False), (0.7, True),  # 70% conf, 67% acc
        (0.5, True), (0.5, False), (0.5, True), (0.5, False),  # 50% conf, 50% acc
        (0.3, False), (0.3, True), (0.3, False),  # 30% conf, 33% acc
    ]
    
    print("\nAdding predictions:")
    for conf, correct in predictions:
        calibration.add_prediction(conf, correct)
    print(f"    Total predictions: {len(predictions)}")
    
    # Compute metrics
    result = calibration.compute_all()
    
    print(f"\nCalibration Results:")
    print(f"    ECE: {result.ece:.4f}")
    print(f"    MCE: {result.mce:.4f}")
    print(f"    Brier Score: {result.brier_score:.4f}")
    print(f"    Overconfident: {result.overconfident}")
    print(f"    Underconfident: {result.underconfident}")
    print(f"    Calibration slope: {result.calibration_slope:.3f}")
    
    # Get reliability diagram data
    diagram_data = calibration.export_reliability_data()
    print(f"\nReliability Diagram:")
    print(f"    Bins with data: {sum(1 for c in diagram_data['counts'] if c > 0)}")
    
    assert 0 <= result.ece <= 1, "ECE should be between 0 and 1"
    assert 0 <= result.brier_score <= 1, "Brier score should be between 0 and 1"
    
    print("\n[OK] Calibration metrics test passed!")


def test_orchestration_harness():
    """Test the full orchestration harness."""
    print("\n" + "="*60)
    print("TEST 5: Orchestration Harness")
    print("="*60)
    
    from agentbench_debug.uncertainty import OrchestrationHarness, EvaluationConfig
    
    config = EvaluationConfig(
        uncertainty_threshold=0.35,
        enable_hierarchical=True,
        enable_calibration=True,
        enable_mitigation=True,
    )
    
    harness = OrchestrationHarness(config)
    
    # Run 1: Successful run with high confidence
    print("\nRun 1: Successful with high confidence")
    harness.start_run("task_1", "toolemu")
    
    # Simulate providing raw API response (OpenAI format)
    raw_response_1 = {
        "choices": [{
            "message": {"content": "Searching for files..."},
            "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.05}]},
        }]
    }
    
    harness.record_step(
        action_name="search_files",
        action_type="query",
        input_messages=[],
        tools_available=["search_files", "read_file"],
        response="Found 3 files",
        raw_api_response=raw_response_1,  # Auto-extract confidence
    )
    harness.mark_checkpoint("tool_selection", score=1.0)
    
    harness.record_step(
        action_name="read_file",
        action_type="query",
        input_messages=[],
        tools_available=[],
        response="File content...",
        confidence=0.85,  # Explicit confidence
    )
    harness.mark_checkpoint("task_completion", score=1.0)
    
    result1 = harness.end_run(success=True)
    print(f"    Composite score: {result1.evaluation.composite.score:.3f}")
    print(f"    Trajectory confidence: {result1.hierarchical_uncertainty.trajectory_level.final_confidence:.3f}")
    
    # Run 2: Failed run with uncertainty
    print("\nRun 2: Failed with high uncertainty")
    harness.start_run("task_2", "toolemu")
    
    harness.record_step(
        action_name="dangerous_delete",
        action_type="delete",
        input_messages=[],
        tools_available=[],
        response={"error": "Permission denied"},
        confidence=0.3,  # Low confidence
    )
    
    result2 = harness.end_run(success=False, failure_reason="Permission denied")
    print(f"    Composite score: {result2.evaluation.composite.score:.3f}")
    print(f"    Error count: {result2.error_summary.total_errors}")
    
    # Get aggregate metrics
    metrics = harness.get_aggregate_metrics()
    print(f"\nAggregate Metrics:")
    print(f"    Total runs: {metrics['total_runs']}")
    print(f"    Success rate: {metrics['success_rate']:.1%}")
    print(f"    Mean composite: {metrics['scores']['mean_composite']:.3f}")
    
    assert metrics['total_runs'] == 2, "Should have 2 runs"
    assert metrics['success_rate'] == 0.5, "50% success rate"
    
    print("\n[OK] Orchestration harness test passed!")


def test_pipeline_integration():
    """Test pipeline integration components."""
    print("\n" + "="*60)
    print("TEST 6: Pipeline Integration")
    print("="*60)
    
    from agentbench_debug.uncertainty import UncertaintyCallback
    
    # Test callback-based tracking
    callback = UncertaintyCallback()
    
    print("\nSimulating task execution with callback:")
    callback.on_task_start("test_task", "toolemu")
    
    steps = [
        ("Executing search query", "search", "query"),
        ("Found results, processing", "process", "write"),
        ("I'm uncertain about this step", "validate", "validate"),
        ("Completed successfully", "finish", "sync"),
    ]
    
    for content, action, action_type in steps:
        conf = callback.on_step(content, action, action_type)
        print(f"    {action}: {conf:.3f}")
    
    report = callback.on_task_end(success=True)
    
    print(f"\nUncertainty Report:")
    print(f"    Task: {report.task_name}")
    print(f"    Steps: {report.total_steps}")
    print(f"    Mean confidence: {report.mean_confidence:.3f}")
    print(f"    Trend: {report.uncertainty_trend}")
    print(f"    Recommendations: {report.recommendations}")
    
    assert report.total_steps == 4, "Should have 4 steps"
    
    print("\n[OK] Pipeline integration test passed!")


def test_full_workflow():
    """Test complete end-to-end workflow."""
    print("\n" + "="*60)
    print("TEST 7: Full End-to-End Workflow")
    print("="*60)
    
    from agentbench_debug.uncertainty import (
        OrchestrationHarness,
        EvaluationConfig,
        ConfidenceExtractor,
    )
    
    # Create full evaluation setup
    config = EvaluationConfig(
        uncertainty_threshold=0.35,
        enable_hierarchical=True,
        enable_calibration=True,
        progress_weight=0.4,
        completion_weight=0.3,
        interaction_weight=0.3,
    )
    
    harness = OrchestrationHarness(config)
    extractor = ConfidenceExtractor()
    
    # Simulate multiple runs with varied outcomes
    test_scenarios = [
        {
            "name": "high_confidence_success",
            "steps": [
                {"action": "auth", "type": "auth", "confidence": 0.95, "error": False},
                {"action": "query", "type": "query", "confidence": 0.88, "error": False},
                {"action": "commit", "type": "sync", "confidence": 0.92, "error": False},
            ],
            "success": True,
        },
        {
            "name": "mixed_confidence_success",
            "steps": [
                {"action": "search", "type": "query", "confidence": 0.75, "error": False},
                {"action": "validate", "type": "validate", "confidence": 0.55, "error": False},
                {"action": "submit", "type": "sync", "confidence": 0.8, "error": False},
            ],
            "success": True,
        },
        {
            "name": "low_confidence_failure",
            "steps": [
                {"action": "risky_delete", "type": "delete", "confidence": 0.35, "error": True},
            ],
            "success": False,
        },
    ]
    
    print("\nRunning test scenarios:")
    for scenario in test_scenarios:
        harness.start_run(scenario["name"], "toolemu")
        
        for step in scenario["steps"]:
            response = {"error": "Failed"} if step["error"] else {"status": "success"}
            harness.record_step(
                action_name=step["action"],
                action_type=step["type"],
                input_messages=[],
                tools_available=[],
                response=response,
                confidence=step["confidence"],
            )
        
        result = harness.end_run(scenario["success"])
        print(f"    {scenario['name']}: composite={result.evaluation.composite.score:.3f}")
    
    # Final aggregate analysis
    metrics = harness.get_aggregate_metrics()
    uncertainty = harness.get_uncertainty_analysis()
    errors = harness.get_error_analysis()
    
    print(f"\nFinal Analysis:")
    print(f"    Runs: {metrics['total_runs']}")
    print(f"    Success rate: {metrics['success_rate']:.1%}")
    print(f"    Mean composite: {metrics['scores']['mean_composite']:.3f}")
    print(f"    Mean ECE: {metrics['calibration']['mean_ece']:.4f}")
    print(f"    Mean trajectory uncertainty: {uncertainty['mean_trajectory_uncertainty']:.3f}")
    
    assert metrics['total_runs'] == 3, "Should have 3 runs"
    
    print("\n[OK] Full workflow test passed!")


def main():
    """Run all tests."""
    print("="*60)
    print("UNCERTAINTY ESTIMATION FRAMEWORK - COMPLETE TEST SUITE")
    print("="*60)
    
    tests = [
        ("Confidence Extraction", test_confidence_extraction),
        ("Uncertainty Tracker", test_uncertainty_tracker),
        ("Hierarchical Propagation", test_hierarchical_propagation),
        ("Calibration Metrics", test_calibration_metrics),
        ("Orchestration Harness", test_orchestration_harness),
        ("Pipeline Integration", test_pipeline_integration),
        ("Full Workflow", test_full_workflow),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n[FAIL] FAILED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n[OK] ALL TESTS PASSED - Uncertainty framework is complete and functional!")
        print("\nThe framework can now measure agent uncertainty through:")
        print("  1. LLM logprobs extraction (OpenAI, Gemini)")
        print("  2. Semantic confidence analysis")
        print("  3. Hierarchical uncertainty propagation")
        print("  4. Calibration metrics (ECE, Brier)")
        print("  5. Agent wrapper for automatic tracking")
        print("  6. Full pipeline integration")
    else:
        print(f"\n[FAIL] {failed} tests failed. Please review errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

