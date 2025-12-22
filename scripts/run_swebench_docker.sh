#!/bin/bash
# Run SWE-bench with Docker-based code exploration

set -e

echo "=========================================="
echo "SWE-bench Docker Task Runner"
echo "=========================================="
echo ""

# Check if .env exists and load it
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check API key
if [ -z "$AZURE_OPENAI_API_KEY" ]; then
    echo "ERROR: AZURE_OPENAI_API_KEY not set!"
    echo "Please set it in .env file or export it:"
    echo "  export AZURE_OPENAI_API_KEY='your-key-here'"
    exit 1
fi

echo "✓ Azure OpenAI API key is set"
echo ""

# Run the assigner
echo "Starting SWE-bench Docker task..."
echo "This will:"
echo "  1. Load SWE-bench instances"
echo "  2. Create Docker containers for each instance"
echo "  3. Clone repositories"
echo "  4. Let agents explore code and generate patches"
echo "  5. Save patches to outputs/swebench_docker_predictions_dev.jsonl"
echo ""

python -m src.assigner --config configs/assignments/swebench-docker-test.yaml

echo ""
echo "=========================================="
echo "Task complete!"
echo "Patches saved to: outputs/swebench_docker_predictions_dev.jsonl"
echo ""
echo "Next step: Evaluate with official harness:"
echo "  ./scripts/evaluate_patches.sh outputs/swebench_docker_predictions_dev.jsonl"
echo "=========================================="
