# Uncertainty Estimation Framework

A complete framework for measuring and analyzing agent uncertainty in LLM-based API orchestration tasks.

## Overview

This framework provides comprehensive uncertainty quantification for AgentBench evaluations, implementing:

- **Confidence Extraction** from LLM API responses (OpenAI, Gemini, Anthropic)
- **Hierarchical Uncertainty Propagation** (Token → Action → Observation → Trajectory)
- **Calibration Metrics** (ECE, Brier Score, Reliability Diagrams)
- **Agent Wrappers** for automatic uncertainty tracking
- **Pipeline Integration** for seamless use with AgentBench tasks

## Quick Start

### Option 1: Wrap an Agent (Recommended)

The easiest way to add uncertainty tracking to any agent:

```python
from agentbench_debug.uncertainty import create_uncertainty_agent

# Wrap your existing agent
wrapped_agent = create_uncertainty_agent(
    agent,
    task_type="toolemu",  # or "dbbench", "os", etc.
    api_type="openai",    # or "gemini", "anthropic", "auto"
)

# Use the agent normally
result = task.run(wrapped_agent)

# Get uncertainty analysis
report = wrapped_agent.get_uncertainty_report("my_task", success=True)
report.print_summary()
```

### Option 2: Manual Tracking with UncertaintyTracker

For custom evaluation loops:

```python
from agentbench_debug.uncertainty import UncertaintyTracker

tracker = UncertaintyTracker(uncertainty_threshold=0.35)

# After each agent step
for step in workflow_steps:
    response = agent.inference(messages)
    
    # Record the response and get confidence
    confidence = tracker.record_response(
        content=response,
        action_name="search_files",
        action_type="query",
        raw_response=api_response,  # Optional: raw API response for better extraction
    )
    
    print(f"Step confidence: {confidence:.3f}")

# Get full analysis
analysis = tracker.get_analysis()
print(f"Mean confidence: {analysis['mean_confidence']:.3f}")
print(f"Trend: {analysis['trend']}")
```

### Option 3: Callback-Based Integration

For event-driven pipelines:

```python
from agentbench_debug.uncertainty import UncertaintyCallback

callback = UncertaintyCallback()

# Start task
callback.on_task_start("task_name", "toolemu")

# After each step
for step in steps:
    response = agent.inference(messages)
    confidence = callback.on_step(
        content=response,
        action_name="tool_name",
        action_type="query",
    )

# End task and get report
report = callback.on_task_end(success=True)
report.print_summary()
```

### Option 4: Analyze Saved Runs

For post-hoc analysis of existing results:

```python
from agentbench_debug.uncertainty import analyze_saved_runs

# Analyze a runs.jsonl file
results = analyze_saved_runs(
    "outputs/toolemu_gemini/runs.jsonl",
    task_type="toolemu",
)

print(f"Success rate: {results['metrics']['success_rate']:.1%}")
print(f"Mean ECE: {results['metrics']['calibration']['mean_ece']:.4f}")
```

## Full Orchestration Harness

For comprehensive evaluation with all features:

```python
from agentbench_debug.uncertainty import (
    OrchestrationHarness,
    EvaluationConfig,
)

# Configure evaluation
config = EvaluationConfig(
    uncertainty_threshold=0.35,
    enable_hierarchical=True,
    enable_calibration=True,
    enable_mitigation=True,
    progress_weight=0.4,
    completion_weight=0.3,
    interaction_weight=0.3,
)

harness = OrchestrationHarness(config)

# Process runs
for task in tasks:
    harness.start_run(task.name, "toolemu")
    
    for step in task.steps:
        harness.record_step(
            action_name=step.action,
            action_type=step.type,
            input_messages=step.messages,
            tools_available=step.tools,
            response=step.response,
            # Option A: Provide explicit confidence
            confidence=0.85,
            # Option B: Let framework extract from raw API response
            raw_api_response=step.raw_response,
        )
    
    harness.end_run(task.success)

# Get aggregate metrics
metrics = harness.get_aggregate_metrics()
print(f"Mean composite score: {metrics['scores']['mean_composite']:.3f}")
print(f"Mean ECE: {metrics['calibration']['mean_ece']:.4f}")

# Save results
harness.save_results("outputs/uncertainty_analysis")
```

## Confidence Extraction

The framework can extract confidence from various sources:

### From OpenAI API Responses (with logprobs)

```python
from agentbench_debug.uncertainty import ConfidenceExtractor

extractor = ConfidenceExtractor()

# OpenAI response with logprobs
response = {
    "choices": [{
        "message": {"content": "The answer is 42."},
        "logprobs": {
            "content": [
                {"token": "The", "logprob": -0.05},
                {"token": " answer", "logprob": -0.1},
            ]
        }
    }]
}

signals = extractor.extract(response, api_type="openai")
print(f"Confidence: {signals.confidence:.3f}")
print(f"Source: {signals.source}")  # "logprobs"
print(f"Mean logprob: {signals.mean_logprob:.3f}")
```

### From Text (Semantic Analysis)

```python
# Automatically detects uncertainty phrases
text = "I'm not sure, but I think the answer might be around 42."
signals = extractor.extract(text, api_type="generic")

print(f"Confidence: {signals.confidence:.3f}")  # Lower due to uncertainty phrases
print(f"Uncertainty phrases: {signals.uncertainty_phrases}")
# ['i'm not sure', 'i think', 'might be']
```

### Self-Reported Confidence

```python
text = "I am 85% confident that this is correct."
signals = extractor.extract(text)

print(f"Self-reported: {signals.self_reported_confidence}")  # 0.85
```

## Metrics Explained

### Calibration Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| **ECE** (Expected Calibration Error) | Mean gap between confidence and accuracy | < 0.05 |
| **MCE** (Maximum Calibration Error) | Largest gap across bins | < 0.15 |
| **Brier Score** | Mean squared error of predictions | < 0.1 |

### Scoring Metrics

| Score | Formula | Description |
|-------|---------|-------------|
| **Progress** | Σ(weight × completion) / Σ(weight) | Incremental task completion |
| **Completion** | 0 or 1 | Binary success indicator |
| **Composite** | α×Progress + β×Completion + γ×(Progress×Completion) | Balanced combination |

### Uncertainty Levels

| Confidence | Interpretation |
|------------|----------------|
| > 0.8 | High confidence - proceed normally |
| 0.6 - 0.8 | Moderate confidence - validate outputs |
| 0.4 - 0.6 | Low confidence - consider alternatives |
| < 0.4 | Very low confidence - likely to fail |

## CLI Usage

```bash
# Analyze existing runs
python scripts/run_uncertainty_evaluation.py \
    --input outputs/toolemu_gemini/runs.jsonl \
    --task-type toolemu \
    --output outputs/uncertainty_analysis \
    --verbose

# With custom threshold
python scripts/run_uncertainty_evaluation.py \
    --input outputs/runs.jsonl \
    --uncertainty-threshold 0.4 \
    --enable-mitigation

# Run tests
python scripts/test_uncertainty_framework.py
```

## Module Structure

```
src/agentbench_debug/uncertainty/
├── __init__.py              # Public API exports
├── README.md                # This file
├── confidence_extractor.py  # LLM confidence extraction
├── agent_wrapper.py         # Agent wrapper for tracking
├── pipeline_integration.py  # Pipeline integration utilities
├── hierarchical.py          # SAUP-style uncertainty propagation
├── calibration.py           # ECE, Brier, reliability diagrams
├── error_taxonomy.py        # API orchestration errors
├── scoring.py               # Progress/completion scoring
├── mitigation.py            # Recovery strategies
├── orchestration_harness.py # Unified evaluation interface
└── quantifier.py            # Basic self-consistency
```

## Example Output

```
============================================================
Uncertainty Report: toolemu_task_001
============================================================
Task Type: toolemu
Success: Yes
Steps: 5

CONFIDENCE METRICS:
  Mean Confidence: 0.823
  Min Confidence: 0.650
  Final Confidence: 0.890
  Uncertainty Trend: stable

CALIBRATION:
  ECE: 0.0450 (Good)

SCORES:
  Composite: 0.850
  Progress: 0.920

RECOMMENDATIONS:
  • Confidence levels acceptable - proceed with standard validation
============================================================
```

## Requirements

- Python 3.8+
- numpy

## References

- SAUP (Situational Awareness Uncertainty Propagation)
- Expected Calibration Error (Guo et al., 2017)
- Circuit Breaker Pattern (Microsoft/Netflix)

---

*Part of the API-ORCHA-Bench uncertainty estimation framework for evaluating LLM-based agents on API orchestration reliability.*

