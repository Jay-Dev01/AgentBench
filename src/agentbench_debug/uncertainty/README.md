# Uncertainty Estimation Framework for AgentBench FC

A complete framework for measuring and analyzing agent uncertainty in LLM-based agent tasks.

## Overview

This framework provides comprehensive uncertainty quantification for AgentBench FC (Function Calling) evaluations, implementing:

- **Confidence Extraction** from LLM API responses (OpenAI, Gemini, Anthropic)
- **Hierarchical Uncertainty Propagation** (Token → Action → Observation → Trajectory)
- **Calibration Metrics** (ECE, Brier Score, Reliability Diagrams)
- **Agent Wrappers** for automatic uncertainty tracking
- **Pipeline Integration** for seamless use with AgentBench FC tasks

## Supported AgentBench FC Tasks

| Task | Code | Description | Function Calls |
|------|------|-------------|----------------|
| **ALFWorld** | `alfworld-std` | Household tasks | `take_action` |
| **DBBench** | `dbbench-std` | Database SQL queries | `execute_sql`, `commit_final_answer` |
| **OS Interaction** | `os-std` | Linux shell tasks | `bash_action`, `finish_action`, `answer_action` |
| **Knowledge Graph** | `kg-std` | Freebase QA | SPARQL queries |
| **WebShop** | `webshop-std` | E-commerce navigation | `search_action`, `click_action` |

## Quick Start

### Option 1: Run with Uncertainty Tracking (Recommended)

Use the uncertainty-aware assigner to run benchmarks with automatic confidence tracking:

```bash
# Run ALFWorld with uncertainty tracking
python -m src.uncertainty_assigner --config configs/assignments/agentbench-fc.yaml

# Results saved to outputs/{TIMESTAMP}/
# - runs.jsonl: Standard results + uncertainty data
# - uncertainty_analysis.json: Aggregate uncertainty metrics
```

### Option 2: Wrap an Agent Programmatically

```python
from agentbench_debug.uncertainty import create_uncertainty_agent

# Wrap your existing agent
wrapped_agent = create_uncertainty_agent(
    agent,
    task_type="alfworld",  # or "dbbench", "os", "kg", "webshop"
    api_type="openai",     # or "gemini", "anthropic", "auto"
)

# Use the agent normally
result = task.run(wrapped_agent)

# Get uncertainty analysis
report = wrapped_agent.get_uncertainty_report("my_task", success=True)
report.print_summary()
```

### Option 3: Manual Tracking with UncertaintyTracker

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
        action_name="take_action",
        action_type="environment_action",
        raw_response=api_response,  # Optional: raw API response
    )
    
    print(f"Step confidence: {confidence:.3f}")

# Get full analysis
analysis = tracker.get_analysis()
print(f"Mean confidence: {analysis['mean_confidence']:.3f}")
print(f"Trend: {analysis['trend']}")
```

### Option 4: Callback-Based Integration

For event-driven pipelines:

```python
from agentbench_debug.uncertainty import UncertaintyCallback

callback = UncertaintyCallback()

# Start task
callback.on_task_start("alfworld_task_001", "alfworld")

# After each step
for step in steps:
    response = agent.inference(messages)
    confidence = callback.on_step(
        content=response,
        action_name="take_action",
        action_type="environment_action",
    )

# End task and get report
report = callback.on_task_end(success=True)
report.print_summary()
```

### Option 5: Analyze Saved Runs

For post-hoc analysis of existing results:

```python
from agentbench_debug.uncertainty import analyze_saved_runs

# Analyze a runs.jsonl file
results = analyze_saved_runs(
    "outputs/gpt-4o-mini/alfworld-std/runs.jsonl",
    task_type="alfworld",
)

print(f"Success rate: {results['metrics']['success_rate']:.1%}")
print(f"Mean ECE: {results['metrics']['calibration']['mean_ece']:.4f}")
```

Or use the CLI script:

```bash
# Analyze all runs in outputs/
python scripts/analyze_real_runs.py --all

# Analyze specific task
python scripts/analyze_real_runs.py --output outputs/2025-12-08-16-44-37 --task alfworld-std
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
    harness.start_run(task.name, "alfworld")
    
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
        "message": {"content": "I'll take the apple."},
        "logprobs": {
            "content": [
                {"token": "I'll", "logprob": -0.05},
                {"token": " take", "logprob": -0.1},
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
text = "I'm not sure, but I think I should go to the kitchen."
signals = extractor.extract(text, api_type="generic")

print(f"Confidence: {signals.confidence:.3f}")  # Lower due to uncertainty phrases
print(f"Uncertainty phrases: {signals.uncertainty_phrases}")
# ['i'm not sure', 'i think']
```

### Self-Reported Confidence

```python
text = "I am 85% confident this is the right path."
signals = extractor.extract(text)

print(f"Self-reported: {signals.self_reported_confidence}")  # 0.85
```

## Task-Specific Action Types

The framework automatically infers action types for AgentBench FC tasks:

| Task | Action | Action Type |
|------|--------|-------------|
| ALFWorld | `take_action` | `environment_action` |
| DBBench | `execute_sql` | `query` |
| DBBench | `commit_final_answer` | `submit` |
| OS | `bash_action` | `shell_command` |
| OS | `finish_action` | `complete` |
| OS | `answer_action` | `submit` |
| WebShop | `search_action` | `search` |
| WebShop | `click_action` | `navigation` |

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
# Run with uncertainty tracking
python -m src.uncertainty_assigner --config configs/assignments/agentbench-fc.yaml

# Analyze existing runs
python scripts/analyze_real_runs.py --all

# Run the test suite
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
Uncertainty Report: alfworld_task_001
============================================================
Task Type: alfworld
Success: Yes
Steps: 12

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

## Running Benchmarks

### Prerequisites

1. Start Docker services:
```bash
cd extra
docker compose up -d controller redis alfworld-std
```

2. Set your API key:
```bash
export OPENAI_API_KEY="your-key-here"
# or for Azure:
export AZURE_OPENAI_API_KEY="your-azure-key-here"
```

### Run with Uncertainty Tracking

```bash
# Run ALFWorld (default)
python -m src.uncertainty_assigner

# Run with specific config
python -m src.uncertainty_assigner --config configs/assignments/agentbench-fc.yaml
```

### Analyze Results

```bash
# View uncertainty analysis
cat outputs/{TIMESTAMP}/uncertainty_analysis.json | python -m json.tool

# Or use the analysis script
python scripts/analyze_real_runs.py --output outputs/{TIMESTAMP}
```

## Requirements

- Python 3.9+
- numpy
- Docker (for task environments)

## References

- AgentBench FC (Function Calling) - THUDM/AgentBench
- SAUP (Situational Awareness Uncertainty Propagation)
- Expected Calibration Error (Guo et al., 2017)
- Circuit Breaker Pattern (Microsoft/Netflix)

---

*Part of the AgentBench FC uncertainty estimation framework for evaluating LLM-based agents on function-calling tasks.*
