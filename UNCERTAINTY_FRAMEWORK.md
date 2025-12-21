# API-ORCHA-Bench Uncertainty Estimation Framework

This document describes the uncertainty estimation framework implemented for the AgentBench project, based on the API-ORCHA-Bench research proposal.

## Overview

The framework provides comprehensive uncertainty quantification for LLM-based agents performing multi-step API orchestration tasks. It implements:

1. **Hierarchical Uncertainty Propagation** (SAUP-style)
2. **Calibration Metrics** (ECE, Brier Score)
3. **API Orchestration Error Taxonomy**
4. **Dual Scoring System** (Progress + Completion)
5. **Mitigation Strategies** (Retry, Circuit Breaker, Validation)

## Quick Start

### 1. Run Uncertainty Evaluation on Existing Results

```bash
# Navigate to AgentBench directory
cd AgentBench

# Run evaluation on ToolEmu results
python scripts/run_uncertainty_evaluation.py \
    --input outputs/toolemu_gemini/runs.jsonl \
    --task-type toolemu \
    --output outputs/uncertainty_analysis

# Run on DBBench results
python scripts/run_uncertainty_evaluation.py \
    --input outputs/dbbench/runs.jsonl \
    --task-type dbbench

# With custom threshold
python scripts/run_uncertainty_evaluation.py \
    --input outputs/runs.jsonl \
    --uncertainty-threshold 0.4 \
    --enable-mitigation
```

### 2. Programmatic Usage

```python
from agentbench_debug.uncertainty import (
    OrchestrationHarness,
    EvaluationConfig,
    HierarchicalUncertaintyPropagator,
    CalibrationMetrics,
    ErrorTaxonomy,
    ScoringSystem,
)

# Configure evaluation
config = EvaluationConfig(
    uncertainty_threshold=0.35,
    enable_hierarchical=True,
    enable_calibration=True,
    progress_weight=0.4,
    completion_weight=0.3,
    interaction_weight=0.3,
)

# Create harness
harness = OrchestrationHarness(config)

# Start a run
harness.start_run("task_1", "toolemu")

# Record steps
harness.record_step(
    action_name="search_users",
    action_type="query",
    input_messages=[{"role": "user", "content": "Find users"}],
    tools_available=["search_users", "delete_user"],
    response={"users": [...]},
    confidence=0.85,
)

# Mark checkpoints
harness.mark_checkpoint("tool_selection", score=1.0)

# End run
result = harness.end_run(success=True)

# Get aggregate metrics
metrics = harness.get_aggregate_metrics()
```

## Component Details

### 1. Hierarchical Uncertainty Propagation

Located in: `src/agentbench_debug/uncertainty/hierarchical.py`

Implements SAUP-style multi-level uncertainty tracking:

```python
from agentbench_debug.uncertainty import HierarchicalUncertaintyPropagator

propagator = HierarchicalUncertaintyPropagator(
    uncertainty_threshold=0.35,
    enable_hmm=True,
)

# Token-level (from LLM logprobs)
token_unc = propagator.compute_token_uncertainty(logprobs=[-0.1, -0.5, -0.2])

# Action-level
action_unc = propagator.compute_action_uncertainty(
    action_name="delete_file",
    action_type="delete",
    confidence=0.7,
    alternatives=[("backup_file", 0.2), ("move_file", 0.1)],
)

# Observation-level
obs_unc = propagator.compute_observation_uncertainty(
    response={"status": "success", "data": {...}},
    expected_schema={"required": ["status", "data"]},
)

# Trajectory-level
trajectory_unc = propagator.compute_trajectory_uncertainty(trajectory_id="run_001")

# Complete analysis
result = propagator.analyze_complete()
print(f"Aggregated confidence: {result.aggregated_score}")
```

**Criticality Weights:**
- Authentication actions: 2.0 (high weight)
- Delete operations: 2.0 (high weight)
- Write/modify: 1.5
- Read/query: 0.8
- Validation: 1.0
- Sync operations: 1.2

### 2. Calibration Metrics

Located in: `src/agentbench_debug/uncertainty/calibration.py`

Measures alignment between predicted confidence and actual outcomes:

```python
from agentbench_debug.uncertainty import CalibrationMetrics

calibration = CalibrationMetrics(n_bins=10)

# Add prediction-outcome pairs
calibration.add_prediction(confidence=0.9, correct=True)
calibration.add_prediction(confidence=0.8, correct=True)
calibration.add_prediction(confidence=0.7, correct=False)

# Compute metrics
result = calibration.compute_all()

print(f"ECE: {result.ece}")          # Lower is better
print(f"Brier Score: {result.brier_score}")
print(f"Overconfident: {result.overconfident}")
print(f"Underconfident: {result.underconfident}")

# Get data for reliability diagram
diagram_data = calibration.export_reliability_data()
```

**Metrics:**
- **ECE (Expected Calibration Error)**: Mean gap between confidence and accuracy
- **MCE (Maximum Calibration Error)**: Largest gap across bins
- **Brier Score**: Mean squared error between confidence and outcome
- **Reliability Diagram**: Visual calibration assessment

### 3. Error Taxonomy

Located in: `src/agentbench_debug/uncertainty/error_taxonomy.py`

Classifies and tracks API orchestration errors:

```python
from agentbench_debug.uncertainty import ErrorTaxonomy, ErrorCategory

taxonomy = ErrorTaxonomy()

# Log an error
error = taxonomy.log_error(
    step_idx=3,
    action_name="authenticate",
    message="Token expired",
    http_status=401,
    uncertainty=0.6,
)

print(f"Category: {error.category}")  # ErrorCategory.AUTHENTICATION
print(f"Type: {error.error_type}")    # "token_expired"
print(f"Recoverable: {error.recoverable}")

# Get recovery strategies
strategies = taxonomy.get_recovery_strategies("token_expired")
# Returns: ["refresh_token", "re_authenticate", "use_cached_credentials"]

# Mark recovery
taxonomy.mark_recovery(error.error_id, "refresh_token", successful=True)

# Get summary
summary = taxonomy.compute_summary()
```

**Error Categories:**
1. **Authentication**: Invalid credentials, token expiry, OAuth violations
2. **Rate Limiting**: Quota exceeded, burst limits, improper backoff
3. **Partial Failure**: Timeouts, partial responses, cascade failures
4. **Data Consistency**: Schema validation, conflicts, stale data
5. **Coordination**: Sequence violations, race conditions, dependencies

### 4. Scoring System

Located in: `src/agentbench_debug/uncertainty/scoring.py`

Implements the dual scoring metrics from the proposal:

```python
from agentbench_debug.uncertainty import ScoringSystem, CheckpointType

scoring = ScoringSystem(
    alpha=0.4,   # Progress weight
    beta=0.3,    # Completion weight
    gamma=0.3,   # Interaction weight
)

# Add checkpoints
scoring.add_checkpoint("auth", CheckpointType.AUTHENTICATION, "Authenticate", weight=1.5)
scoring.add_checkpoint("query", CheckpointType.API_COORDINATION, "Query data", weight=1.0)
scoring.add_checkpoint("validate", CheckpointType.DATA_CONSISTENCY, "Validate", weight=1.2)

# Mark progress
scoring.mark_checkpoint_completed("auth", partial_score=1.0)
scoring.mark_checkpoint_completed("query", partial_score=0.8)

# Record steps with uncertainty
scoring.add_step(uncertainty=0.2)
scoring.add_step(uncertainty=0.4)

# Mark completion
scoring.mark_workflow_complete(success=True)

# Get scores
progress = scoring.compute_progress_score()
completion = scoring.compute_full_completion_score()
composite = scoring.compute_composite_score()

print(f"Progress: {progress.score}")
print(f"Completion: {completion.score}")
print(f"Composite: {composite.score}")
print(f"Interpretation: {composite.interpretation}")
```

**Scoring Formula:**
```
S_composite = α × S_progress + β × S_full + γ × S_progress × S_full
```

Where the interaction term `γ × S_progress × S_full` rewards high-progress completions.

### 5. Mitigation Strategies

Located in: `src/agentbench_debug/uncertainty/mitigation.py`

Implements the Decision Layer for adaptive recovery:

```python
from agentbench_debug.uncertainty import (
    DecisionLayer,
    UncertaintyAwareRetry,
    CircuitBreaker,
    CrossAPIValidator,
)

# Use decision layer for automatic strategy selection
decision_layer = DecisionLayer()

# Select strategies based on context
strategies = decision_layer.select_strategy(
    uncertainty=0.7,
    error_type="quota_exceeded",
    error_category="rate_limiting",
    previous_strategies=["retry_with_backoff"],
)
# Returns: [RATE_LIMITER, CIRCUIT_BREAKER]

# Execute a strategy
success, result = decision_layer.execute_strategy(
    MitigationStrategy.RETRY_WITH_BACKOFF,
    context={
        "operation": lambda: api_call(),
        "uncertainty": 0.5,
    }
)

# Get strategy statistics
stats = decision_layer.get_strategy_stats()
```

**Available Strategies:**
- **RETRY_WITH_BACKOFF**: Uncertainty-aware exponential backoff
- **CROSS_API_VALIDATION**: Consistency checking across APIs
- **CIRCUIT_BREAKER**: Failure isolation pattern
- **RATE_LIMITER**: Adaptive request pacing
- **FALLBACK_SERVICE**: Alternative endpoint usage
- **GRACEFUL_DEGRADATION**: Partial/cached results

### 6. Orchestration Harness

Located in: `src/agentbench_debug/uncertainty/orchestration_harness.py`

Unified interface integrating all components:

```python
from agentbench_debug.uncertainty import OrchestrationHarness, EvaluationConfig

config = EvaluationConfig(
    uncertainty_threshold=0.35,
    enable_hierarchical=True,
    enable_calibration=True,
    enable_mitigation=True,
)

harness = OrchestrationHarness(config)

# Process multiple runs
for task in tasks:
    harness.start_run(task.name, task.type)
    
    for step in task.steps:
        harness.record_step(
            action_name=step.action,
            action_type=step.type,
            input_messages=step.messages,
            tools_available=step.tools,
            response=step.response,
            confidence=step.confidence,
        )
    
    harness.end_run(task.success)

# Get aggregate metrics
metrics = harness.get_aggregate_metrics()
error_analysis = harness.get_error_analysis()
uncertainty_analysis = harness.get_uncertainty_analysis()

# Save results
harness.save_results("outputs/analysis")
```

## Output Formats

### JSON Report
```json
{
  "metrics": {
    "total_runs": 100,
    "success_rate": 0.85,
    "scores": {
      "mean_progress": 0.78,
      "mean_completion": 0.85,
      "mean_composite": 0.81
    },
    "calibration": {
      "mean_ece": 0.045,
      "mean_brier": 0.12
    }
  },
  "error_analysis": {
    "errors_by_category": {...},
    "top_10_errors": [...]
  },
  "uncertainty_analysis": {
    "mean_trajectory_uncertainty": 0.32,
    "trend_distribution": {...}
  }
}
```

### Markdown Report
Generated automatically with tables and visualizable metrics.

## Integration with AgentBench Tasks

### ToolEmu
```python
harness.start_run("toolemu_test", "toolemu")
harness.scoring.setup_toolemu_checkpoints()
# Checkpoints: tool_selection, parameter_validation, safety_compliance, task_completion
```

### DBBench
```python
harness.start_run("dbbench_test", "dbbench")
harness.scoring.setup_dbbench_checkpoints()
# Checkpoints: query_formulation, syntax_valid, semantic_correct, result_interpretation
```

### Custom API Orchestration
```python
harness.start_run("api_workflow", "custom")
harness.scoring.setup_api_orchestration_checkpoints(
    n_apis=3,
    include_auth=True,
    include_recovery=True,
)
```

## Running Tests

```bash
# First, run AgentBench tasks to generate results
docker compose -f extra/docker-compose.yml up -d controller dbbench-std toolemu-std
python -m src.assigner --config configs/assignments/toolemu-gemini.yaml

# Then analyze results
python scripts/run_uncertainty_evaluation.py \
    --input outputs/toolemu_gemini/runs.jsonl \
    --task-type toolemu \
    --output outputs/uncertainty_analysis \
    --verbose

# View the generated report
cat outputs/uncertainty_analysis/report_*.md
```

## Ideal Results (from Research Proposal)

Per the API-ORCHA-Bench proposal, ideal results would show:

1. **Composite Scores ≥ 0.85** for top-performing agents
2. **ECE Reduction from 0.15 to < 0.05** with SAUP enhancement
3. **30-40% improvement** in uncertainty calibration vs. baseline
4. **Strong correlation** between predicted confidence and actual success
5. **Intelligent degradation patterns**: High uncertainty → appropriate fallbacks

## File Structure

```
src/agentbench_debug/uncertainty/
├── __init__.py              # Module exports
├── quantifier.py            # Basic self-consistency quantifier
├── hierarchical.py          # SAUP-style multi-level uncertainty
├── calibration.py           # ECE, Brier, reliability diagrams
├── error_taxonomy.py        # API orchestration error classification
├── scoring.py               # Progress, completion, composite scores
├── mitigation.py            # Retry, circuit breaker, validation
└── orchestration_harness.py # Unified evaluation interface

scripts/
└── run_uncertainty_evaluation.py  # CLI runner
```

## References

- SAUP (Situational Awareness Uncertainty Propagation)
- AgentDebug error taxonomy
- Circuit Breaker pattern (Microsoft/Netflix)
- Expected Calibration Error (Guo et al., 2017)

---

*This framework implements the uncertainty estimation components of the API-ORCHA-Bench research proposal for evaluating LLM-based agents on API orchestration reliability.*

