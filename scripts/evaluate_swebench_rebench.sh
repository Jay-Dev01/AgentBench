#!/bin/bash
# Evaluate agent-generated patches with official SWE-bench harness

PREDICTIONS_FILE=${1:-outputs/swebench_rebench_dev_predictions.jsonl}
RUN_ID=${2:-agentbench-$(date +%Y%m%d-%H%M%S)}
MAX_WORKERS=${3:-4}

echo "=========================================="
echo "SWE-bench Evaluation"
echo "=========================================="
echo "Predictions: $PREDICTIONS_FILE"
echo "Run ID: $RUN_ID"
echo "Max Workers: $MAX_WORKERS"
echo ""

if [ ! -f "$PREDICTIONS_FILE" ]; then
    echo "Error: Predictions file not found: $PREDICTIONS_FILE"
    exit 1
fi

NUM_PREDICTIONS=$(wc -l < "$PREDICTIONS_FILE")
echo "Found $NUM_PREDICTIONS predictions to evaluate"
echo ""

# Check if SWE-bench harness is installed
if ! python -c "import swebench" 2>/dev/null; then
    echo "Error: SWE-bench harness not installed"
    echo "Please install it first:"
    echo "  cd path/to/swe-bench && pip install -e ."
    exit 1
fi

# Run evaluation
python -m swebench.harness.run_evaluation \
    --dataset_name nebius/SWE-rebench \
    --predictions_path "$PREDICTIONS_FILE" \
    --max_workers "$MAX_WORKERS" \
    --run_id "$RUN_ID" \
    --namespace ""

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results in: logs/run_evaluation/$RUN_ID/"
echo "=========================================="
