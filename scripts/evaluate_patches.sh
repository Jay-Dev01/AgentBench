#!/bin/bash
# Evaluate agent-generated patches with official SWE-bench harness

set -e

PREDICTIONS_FILE=${1:-outputs/swebench_docker_predictions_dev.jsonl}
RUN_ID=${2:-agentbench-docker-$(date +%Y%m%d-%H%M%S)}
MAX_WORKERS=${3:-4}

echo "=========================================="
echo "SWE-bench Official Harness Evaluation"
echo "=========================================="
echo "Predictions file: $PREDICTIONS_FILE"
echo "Run ID: $RUN_ID"
echo "Max workers: $MAX_WORKERS"
echo ""

# Check if predictions file exists
if [ ! -f "$PREDICTIONS_FILE" ]; then
    echo "ERROR: Predictions file not found: $PREDICTIONS_FILE"
    echo ""
    echo "Please run the task first:"
    echo "  ./scripts/run_swebench_docker.sh"
    exit 1
fi

# Count predictions
NUM_PREDICTIONS=$(wc -l < "$PREDICTIONS_FILE" | tr -d ' ')
echo "Found $NUM_PREDICTIONS predictions to evaluate"
echo ""

# Ask for confirmation if many predictions
if [ "$NUM_PREDICTIONS" -gt 10 ]; then
    echo "WARNING: Evaluating $NUM_PREDICTIONS instances will take significant time and resources."
    echo "Each instance requires building a Docker image and running tests."
    echo ""
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 1
    fi
fi

echo "Starting evaluation..."
echo "This may take a while. Docker images will be built and tests will run."
echo ""

# Run evaluation using official SWE-bench harness
cd swe-bench-reference

python -m swebench.harness.run_evaluation \
    --dataset_name nebius/SWE-rebench \
    --predictions_path "../$PREDICTIONS_FILE" \
    --max_workers "$MAX_WORKERS" \
    --run_id "$RUN_ID" \
    --namespace ""

cd ..

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo ""
echo "Results saved to: swe-bench-reference/logs/run_evaluation/$RUN_ID/"
echo ""
echo "To view results:"
echo "  cat swe-bench-reference/logs/run_evaluation/$RUN_ID/report.json"
echo "=========================================="
