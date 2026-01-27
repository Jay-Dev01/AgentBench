# Uncertainty Estimation Framework for AgentBench

This module provides comprehensive uncertainty quantification for LLM agent workflows in AgentBench. It supports both real-time tracking during agent execution and post-hoc analysis of saved runs.

## Features

### 1. Enhanced Confidence Extraction
Extract confidence from various LLM API response formats:
- **Logprobs**: Token-level probabilities (geometric mean, Eq. 1 in paper)
- **finish_reason**: `stop` → 0.85, `tool_calls` → 0.80, `length` → 0.50
- **Tool call analysis**: Semantic analysis of tool arguments for hedging
- **Command patterns**: Risk assessment of bash commands (`rm -rf` → lower confidence)
- **Semantic hedging**: Detect phrases like "I think", "probably", "maybe"
- **Confident phrases**: Detect "definitely", "certainly", "this fixes"
- **Self-reported**: Parse statements like "I am 80% confident"

### 2. Hierarchical Uncertainty (SAUP-style)
Multi-level uncertainty propagation:
- **Token level**: Individual token probabilities
- **Action level**: Aggregated action confidence with criticality weights
- **Observation level**: Environment feedback uncertainty
- **Trajectory level**: Overall workflow uncertainty (Utraj = 1 - mean(ct), Eq. 2)

### 3. Calibration Metrics
- **ECE** (Expected Calibration Error)
- **MCE** (Maximum Calibration Error)
- **Brier Score**
- **Spearman ρ**: Correlation between confidence and task success
- **AUROC**: Discrimination ability of confidence scores
- **Outcome Analysis**: Over/underconfidence rates
- **Temperature Scaling** for post-hoc calibration

### 4. Error Taxonomy
Classify and track API orchestration errors:
- Rate limits, authentication, timeouts
- Context limits, content filters
- Network errors, parse errors
- Recovery recommendations for each type

### 5. Scoring System
- Progress score (steps completed)
- Completion score (task success)
- Composite score with configurable weights
- Uncertainty-weighted adjustments

## Quick Start

### Real-Time Tracking (Recommended)

Run AgentBench with automatic uncertainty tracking:

```bash
python -m src.uncertainty_assigner --config configs/assignments/your_config.yaml
```

This will:
1. Wrap all agents with uncertainty tracking
2. Extract confidence from each inference call (with semantic analysis)
3. Compute calibration metrics including Spearman correlation
4. Save analysis to `outputs/<timestamp>/uncertainty_analysis.json`

### Post-Hoc Analysis

Analyze existing results with the analysis script:

```bash
# Analyze most recent run
python scripts/analyze_swebench_results.py --latest

# Analyze specific run
python scripts/analyze_swebench_results.py outputs/2026-01-26-14-28-48

# Output as JSON
python scripts/analyze_swebench_results.py --latest --json
```

This generates a detailed calibration report including:
- Success/failure statistics
- ECE, MCE, Brier Score
- Spearman correlation (confidence-success)
- AUROC discrimination metric
- Over/underconfidence analysis
- Text-based reliability diagram

### Programmatic Usage

```python
from src.agentbench_debug.uncertainty import (
    OrchestrationHarness,
    ConfidenceExtractor,
    UncertaintyAwareAgent,
)
from src.agentbench_debug.uncertainty.calibration import (
    CalibrationMetrics,
    OutcomeAnalysis,
)

# Option 1: Use the orchestration harness
harness = OrchestrationHarness(task_id="task-001", task_type="swebench")
harness.start_workflow(task_id="task-001", task_type="swebench")

for step in agent_steps:
    harness.record_step(
        action=step.action,
        action_type="bash_command",
        observation=step.observation,
        success=step.success,
        raw_api_response=step.raw_response,  # For confidence extraction
    )

result = harness.finish_workflow(task_completed=True)
print(f"Mean confidence: {result.evaluation.mean_confidence}")

# Option 2: Wrap an agent directly
from src.agentbench_debug.uncertainty import wrap_agent

wrapped = wrap_agent(your_agent, task_id="task-001", task_type="swebench")
output = wrapped.inference(history)  # Automatically tracks uncertainty
summary = wrapped.get_uncertainty_summary()

# Option 3: Compute calibration from existing data
calibration_metrics = CalibrationMetrics(n_bins=10)
result = calibration_metrics.compute(confidences, outcomes)
outcome = calibration_metrics.analyze_outcomes(confidences, outcomes)

print(f"ECE: {result.ece:.4f}")
print(f"Spearman ρ: {result.spearman_rho:.4f}")
print(f"AUROC: {result.auroc:.4f}")
print(f"Confidence gap: {outcome.confidence_gap:+.3f}")
```

## Module Structure

```
src/agentbench_debug/uncertainty/
├── __init__.py              # Module exports
├── quantifier.py            # Basic uncertainty from samples
├── confidence_extractor.py  # Enhanced confidence extraction
├── hierarchical.py          # SAUP-style multi-level propagation
├── calibration.py           # ECE, Spearman, AUROC, temperature scaling
├── scoring.py               # Progress and composite scores
├── error_taxonomy.py        # Error classification
├── mitigation.py            # Retry, circuit breaker strategies
├── orchestration_harness.py # Unified interface
├── agent_wrapper.py         # Wrap agents for tracking
├── pipeline_integration.py  # AgentBench integration helpers
└── README.md               # This file

scripts/
├── analyze_swebench_results.py  # Post-hoc calibration analysis
└── test_uncertainty_framework.py # Test suite
```

## Supported Tasks

The framework supports all AgentBench FC tasks:
- **ALFWorld**: Household environment actions
- **DBBench**: SQL database queries
- **OS Interaction**: Shell commands
- **Knowledge Graph**: SPARQL queries
- **WebShop**: E-commerce navigation
- **SWE-bench**: Software engineering patches

## Testing

Run the test suite:

```bash
python scripts/test_uncertainty_framework.py
```

## Output Format

The uncertainty analysis JSON contains:

```json
{
  "summary": {
    "total_runs": 100,
    "successful_runs": 82,
    "failed_runs": 18,
    "success_rate": 0.82,
    "mean_confidence": 0.782,
    "min_confidence": 0.7,
    "max_confidence": 0.8
  },
  "calibration": {
    "ece": 0.029,
    "mce": 0.15,
    "brier_score": 0.121,
    "spearman_rho": 0.35,
    "auroc": 0.72,
    "bin_accuracies": [0.0, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95, 1.0, 1.0],
    "bin_confidences": [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
    "bin_counts": [0, 0, 0, 5, 10, 15, 20, 25, 15, 10]
  },
  "outcome_analysis": {
    "mean_confidence_success": 0.85,
    "mean_confidence_failure": 0.65,
    "confidence_gap": 0.20,
    "spearman_rho": 0.35,
    "auroc": 0.72,
    "overconfidence_rate": 0.25,
    "underconfidence_rate": 0.05
  },
  "interpretation": {
    "ece": "Well-calibrated: confidence closely matches success",
    "spearman": "Good: higher confidence correlates with success",
    "auroc": "Good discrimination: can distinguish success from failure",
    "confidence_gap": "Healthy: agent is more confident when successful"
  },
  "runs": [
    {
      "agent": "gpt-4o-mini",
      "task": "swebench-rebench-100",
      "index": 0,
      "success": true,
      "n_steps": 20,
      "mean_confidence": 0.82,
      "min_confidence": 0.75,
      "trend": "stable",
      "confidence_history": [0.8, 0.85, 0.82, ...]
    }
  ]
}
```

## Configuration

### Confidence Extraction

```python
from src.agentbench_debug.uncertainty import ConfidenceExtractor

# Enable semantic analysis (default: True)
extractor = ConfidenceExtractor(
    default_confidence=0.70,
    use_semantic_analysis=True,
)

# Extract from API response
confidence, source = extractor.extract(raw_response, content)
# Returns e.g.: (0.75, "tool:bash_command+semantic(-0.05)")
```

### Action Complexity Weights

The extractor applies these action-specific adjustments:
- `submit_patch`: 0.75 (higher stakes)
- `bash_command`: 0.80 (command execution)
- `read_file`: 0.90 (low risk)
- `search_code`: 0.85 (exploration)

### Command Pattern Analysis

For `bash_command` actions, the extractor analyzes patterns:

**Risky patterns (lower confidence):**
- `rm -rf`: -0.15
- `sudo`: -0.10
- `kill`: -0.10
- `chmod`: -0.05

**Safe patterns (higher confidence):**
- `ls`, `cat`: +0.05
- `grep`: +0.03
- `pytest -v`: +0.05
- `git status/diff`: +0.05

### Semantic Hedging Detection

The extractor detects uncertainty phrases:
- "I think", "probably", "maybe", "perhaps"
- "not sure", "might be", "could be"
- "try", "attempt", "guess", "assume"

And confidence phrases:
- "definitely", "certainly", "clearly"
- "exactly", "precisely", "this fixes"

## Integration with Paper Methodology

This implementation follows the methodology described in the API-ORCHA-Bench paper:

1. **Token-level confidence** (Eq. 1): Geometric mean of token probabilities
   ```
   ct = exp(1/n * Σ log p(xi | x<i))
   ```

2. **API signal extraction**: `finish_reason` mapping when logprobs unavailable

3. **Trajectory aggregation** (Eq. 2): 
   ```
   Utraj = 1 - (1/T) * Σ ct
   ```

4. **Calibration metrics**:
   - ECE (Expected Calibration Error)
   - Spearman ρ (correlation between Utraj and failure)
   - AUROC (discrimination ability)

5. **Trend tracking**: First-half vs second-half mean comparison

## Limitations and Recommendations

### Current Limitations

1. **Function calling lacks logprobs**: OpenAI doesn't return logprobs for `tool_calls`, so confidence is derived from `finish_reason` (constant 0.80).

2. **Semantic analysis limited**: Tool call arguments may not contain hedging language.

### Recommendations for Better Uncertainty

1. **Implement resampling** (Section 3.3 of paper): Generate multiple trajectories at different temperatures and compare.

2. **Use reasoning models**: Models like o1 that output reasoning may show more hedging.

3. **Add self-consistency checks**: Compare multiple samples for agreement.

4. **Pre-action confidence prompting**: Ask the model to rate its confidence before each action.

## Examples

See `scripts/test_uncertainty_framework.py` for comprehensive examples of all components.

See `scripts/analyze_swebench_results.py` for post-hoc calibration analysis.
