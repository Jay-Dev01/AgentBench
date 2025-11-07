# ToolEmu Integration for AgentBench

This directory contains the integration of ToolEmu's adversarial testing capabilities into AgentBench.

## Overview

ToolEmu provides a comprehensive framework for testing AI agent safety and helpfulness in scenarios with:
- **Underspecified instructions** - Tasks with missing information
- **Adversarial environments** - Stress testing with failure injection
- **Safety-critical scenarios** - Testing for risky actions and unintended consequences
- **Multi-step reasoning** - Complex tasks requiring planning and tool use

## Features

### 1. Adversarial Simulation
The environment wrapper (`environment.py`) supports two modes:
- **Standard Mode**: Realistic tool execution simulation
- **Adversarial Mode**: Stress testing with failure injection including:
  - Partial failures
  - Unexpected data formats
  - Permission errors
  - Timeouts
  - Side effects

### 2. Safety & Helpfulness Evaluation
The evaluator (`evaluator.py`) measures:
- **Safety Score**: Based on avoiding risky actions
- **Helpfulness Score**: Based on achieving expected outcomes
- **Overall Score**: Weighted combination (default: 60% safety, 40% helpfulness)

### 3. Uncertainty Quantification
Placeholder implementation for future research:
- Epistemic uncertainty (model uncertainty)
- Aleatoric uncertainty (environmental uncertainty)
- Decision confidence metrics

## File Structure

```
toolemu/
├── __init__.py           # Package initialization
├── task.py              # Main task class integrating with AgentBench
├── environment.py       # Tool execution simulation with failure injection
├── evaluator.py         # Safety & helpfulness evaluation
├── utils.py             # Utility functions for data loading and conversion
└── README.md           # This file
```

## Configuration

### Task Configurations (configs/tasks/toolemu.yaml)

Four pre-configured task variants:

1. **toolemu-std**: Standard mode with realistic simulation
2. **toolemu-adv**: Adversarial mode with 30% failure injection
3. **toolemu-stress**: High stress mode with 50% failure injection
4. **toolemu-safety**: Safety-focused evaluation (80% safety weight)

### Assignment Configuration (configs/assignments/toolemu.yaml)

Pre-configured to test:
- GPT-4 on all task variants
- GPT-3.5 on standard and adversarial modes

## Usage

### Run ToolEmu Tests

```bash
# From AgentBench root directory
cd AgentBench

# Run with default configuration
python -m src.assigner --config configs/assignments/toolemu.yaml

# Run specific task variant
python -m src.assigner --config configs/assignments/toolemu.yaml --task toolemu-adv
```

### Custom Configuration

You can create custom configurations by modifying the YAML files:

```yaml
toolemu-custom:
  parameters:
    name: toolemu-custom
    data_file: "data/toolemu/all_cases.json"
    toolkits_file: "data/toolemu/all_toolkits.json"
    simulator_type: "adversarial"
    adversarial_probability: 0.4  # Adjust failure injection rate
    safety_weight: 0.7            # Adjust safety importance
    helpfulness_weight: 0.3
    max_round: 20                 # Allow more rounds
```

## Data

### Location
- Test cases: `data/toolemu/all_cases.json` (144 cases)
- Toolkits: `data/toolemu/all_toolkits.json` (38 toolkits, 311+ tools)

### Test Case Structure
Each test case contains:
- **User Instruction**: The task to complete
- **Toolkits**: Required tools
- **Underspecifications**: Missing information that could lead to errors
  - Task Information: Ambiguous task details
  - Safety & Security Constraints: Unstated safety requirements
- **Expected Achievements**: What the agent should accomplish
- **Potential Risky Actions**: Actions that could cause harm
- **Potential Risky Outcomes**: Possible negative consequences

## Evaluation Metrics

### Per-Case Metrics
- `overall_score`: Weighted combination of safety and helpfulness (0-1)
- `passed`: Boolean indicating if the agent passed (safe AND helpful)
- `safety.safety_score`: Safety score (0-1)
- `safety.risky_actions_taken`: List of risky actions performed
- `safety.violations`: List of safety violations
- `helpfulness.helpfulness_score`: Helpfulness score (0-1)
- `helpfulness.achievements_met`: Expected achievements completed
- `execution_summary`: Statistics on tool executions

### Aggregate Metrics (from calculate_overall)
- `pass_rate`: Percentage of cases passed
- `avg_overall_score`: Average overall score across all cases
- `avg_safety_score`: Average safety score
- `avg_helpfulness_score`: Average helpfulness score
- `safety_violation_rate`: Rate of safety violations
- `total_risky_actions`: Total risky actions across all cases

## Research Extensions

This integration is designed to support research on:

1. **Uncertainty Quantification**
   - Extend `UncertaintyQuantifier` class in `evaluator.py`
   - Add model log probability tracking
   - Implement epistemic/aleatoric uncertainty calculation

2. **Failure Injection Strategies**
   - Modify `environment.py` to add new failure scenarios
   - Implement targeted adversarial testing
   - Add domain-specific failure modes

3. **Advanced Evaluation**
   - Integrate LLM-based evaluation for nuanced assessment
   - Add temporal reasoning evaluation
   - Implement multi-agent coordination metrics

4. **Dataset Extension**
   - Add new test cases to `data/toolemu/all_cases.json`
   - Create domain-specific toolkits
   - Generate synthetic test cases

## Implementation Details

### Tool Conversion
ToolEmu tools are automatically converted from their native format to OpenAI function calling format:

```python
# ToolEmu format
{
  "name": "send_email",
  "summary": "Sends an email",
  "parameters": [
    {"name": "recipient", "type": "string", "required": true},
    {"name": "subject", "type": "string", "required": true}
  ]
}

# Converted to OpenAI format
{
  "type": "function",
  "function": {
    "name": "send_email",
    "description": "Sends an email",
    "parameters": {
      "type": "object",
      "properties": {
        "recipient": {"type": "string"},
        "subject": {"type": "string"}
      },
      "required": ["recipient", "subject"]
    }
  }
}
```

### Execution Flow
1. Load test case and required toolkits
2. Convert tools to OpenAI function calling format
3. Initialize environment with simulation mode
4. Inject system prompt and user instruction
5. Run agent-environment interaction loop:
   - Agent generates response (possibly with tool calls)
   - Environment simulates tool execution
   - Results injected back into conversation
6. Evaluate safety and helpfulness
7. Calculate final score and determine pass/fail

## Future Work

- [ ] Implement full LLM-based evaluation (currently keyword-based)
- [ ] Add support for real tool execution (not just simulation)
- [ ] Integrate ToolEmu's original adversarial simulator prompts
- [ ] Add trajectory visualization and analysis tools
- [ ] Implement active learning for test case generation
- [ ] Add support for multi-agent scenarios

## References

- ToolEmu Paper: [Tool Learning with Foundation Models](https://arxiv.org/abs/2304.08354)
- AgentBench: [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688)

## Contact

For questions or issues related to the ToolEmu integration, please refer to the main AgentBench documentation or create an issue in the repository.
