# Testing the Uncertainty Quantification Module

This guide explains how to test and use the new uncertainty quantification features.

## Quick Start: Test with Demo Script

### 1. Create a Test Trajectory File

A sample file `test_trajectory.json` has been created. You can also create your own:

```json
{
  "task": "your-task-id",
  "task_description": "Description of the ALFWorld task",
  "steps": [
    {
      "idx": 0,
      "context": "Current environment context",
      "modules": {"action": "look around"}
    }
  ]
}
```

### 2. Run the Demo Script (Stub LLM)

```bash
# From the AgentBench root directory
python scripts/run_uncertainty_demo.py \
  --input test_trajectory.json \
  --output test_output.json
```

This uses a **stub LLM** that generates random variants. Good for testing the pipeline structure.

### 3. Check the Output

Open `test_output.json` and look for:
- `steps[*].uncertainty.step.score` - uncertainty score (0.0 = confident, 1.0 = very uncertain)
- `steps[*].uncertainty.step.status` - "confident" or "low_confidence"
- `steps[*].uncertainty.details` - detailed breakdown
- `totals.avg_uncertainty` - average across all steps
- `totals.low_conf_steps` - count of steps above threshold

## Using with a Real LLM

### Option 1: Modify the Demo Script

Replace `make_stub_llm` in `run_uncertainty_demo.py` with your actual LLM call:

```python
from openai import OpenAI  # or your LLM client

def make_real_llm():
    client = OpenAI(api_key="your-key")
    
    def _call(prompt: str, temperature: float) -> str:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    
    return _call

# Then in main():
analyzer.set_call_fn(make_real_llm())
```

### Option 2: Use Programmatically

```python
from agentbench_debug.fine_grained.analyzer import FineGrainedAnalyzer

# Initialize analyzer
analyzer = FineGrainedAnalyzer(
    threshold=0.35,  # uncertainty threshold
    k=3,             # number of samples to generate
    temperature=0.7, # sampling temperature
    enable_self_correction=True
)

# Set your LLM function
def my_llm_call(prompt: str, temp: float) -> str:
    # Your LLM implementation here
    return response_text

analyzer.set_call_fn(my_llm_call)

# Analyze a step
result = analyzer.analyze_step(
    step_idx=0,
    task_description="Find the apple",
    context_text="You are in the kitchen...",
    modules={"action": "look around"}
)

print(f"Uncertainty score: {result['uncertainty']['step']['score']}")
print(f"Status: {result['uncertainty']['step']['status']}")
```

## Integration with AgentBench ALFWorld Task

To integrate into the actual ALFWorld task execution:

### 1. Modify `src/server/tasks/alfworld/task.py`

Add uncertainty tracking to the `alfworld_run` method:

```python
from agentbench_debug.fine_grained.analyzer import FineGrainedAnalyzer

class ALFWorld(Task):
    def __init__(self, ...):
        # ... existing init ...
        self.uncertainty_analyzer = FineGrainedAnalyzer(
            threshold=0.35,
            k=3,
            temperature=0.7
        )
        # Set up LLM call function (use your agent's inference method)
        self.uncertainty_analyzer.set_call_fn(self._uncertainty_llm_call)
    
    def _uncertainty_llm_call(self, prompt: str, temperature: float) -> str:
        # Use your agent's inference with temperature
        # This is a placeholder - adapt to your actual agent
        return self.agent.inference([{"role": "user", "content": prompt}], temperature=temperature)
    
    def alfworld_run(self, session: Session, env):
        # ... existing code ...
        
        for i in range(0, self.max_step):
            output = session.sync_action()
            # ... extract action ...
            
            # Add uncertainty analysis
            uncertainty_result = self.uncertainty_analyzer.analyze_step(
                step_idx=i,
                task_description=ob,  # or task description
                context_text=observation,
                modules={"action": action}
            )
            
            # Log or store uncertainty
            log_info["uncertainty"] = uncertainty_result["uncertainty"]
            
            # ... continue with existing code ...
```

### 2. Extract from Real ALFWorld Runs

After running AgentBench, extract trajectories from `runs.jsonl`:

```python
import json

# Read a completed run
with open("outputs/.../alfworld-std/runs.jsonl", "r") as f:
    for line in f:
        run = json.loads(line)
        # Extract steps
        steps = run.get("output", {}).get("steps", [])
        task_desc = run.get("output", {}).get("task_description", "")
        
        # Process with uncertainty analyzer
        # ... use analyzer.analyze_step() for each step ...
```

## Testing Individual Components

### Test UncertaintyQuantifier Directly

```python
from agentbench_debug.uncertainty.quantifier import UncertaintyQuantifier

uq = UncertaintyQuantifier(threshold=0.35)

# Test with samples
samples = [
    "look around",
    "look around", 
    "examine room",  # different
    "look around"
]

result = uq.from_samples(step_idx=0, module="action", samples=samples)
print(result)
# Should show score < 0.35 (mostly agree) or > 0.35 (disagree)
```

### Test Helpers

```python
from agentbench_debug.fine_grained.helpers import (
    extract_action_only,
    regenerate_action_samples
)

# Test action extraction
text = "Some text <action>look around</action> more text"
action = extract_action_only(text)
assert action == "look around"

# Test regeneration (requires LLM configured)
# ... configure call_llm first ...
samples = regenerate_action_samples(
    task_description="Find apple",
    context_text="In kitchen",
    base_action="look around",
    k=3
)
```

## Expected Output Format

The analyzer produces JSON like:

```json
{
  "step_idx": 0,
  "modules": {"action": "look around"},
  "uncertainty": {
    "step": {
      "score": 0.47,
      "status": "low_confidence"
    },
    "details": [{
      "step_idx": 0,
      "module": "action",
      "method": "self_consistency",
      "score": 0.47,
      "status": "low_confidence",
      "meta": {
        "k": 3,
        "threshold": 0.35,
        "samples": ["look around", "examine", "look around"]
      }
    }]
  },
  "self_correction": {
    "fired": true,
    "reason": "uncertainty>threshold",
    "updated_action": "look around carefully"
  }
}
```

## Troubleshooting

1. **Import errors**: Make sure you're running from AgentBench root and `src` is in Python path
2. **No LLM configured**: Use `analyzer.set_call_fn()` before calling `analyze_step()`
3. **Empty samples**: Check that your LLM is returning valid `<action>...</action>` format
4. **High uncertainty on all steps**: This might be expected if actions are genuinely diverse; adjust threshold

## Next Steps

- Run on real ALFWorld trajectories from completed AgentBench runs
- Correlate uncertainty scores with actual errors/failures
- Tune threshold based on empirical results
- Add entropy-based uncertainty (using logprobs) if available

