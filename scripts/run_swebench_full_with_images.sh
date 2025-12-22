#!/bin/bash

# Run SWE-bench Full Dataset with Docker Images
# Usage: ./run_swebench_full_with_images.sh

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=============================================================="
echo "🚀 SWE-bench Full Dataset with Docker Images Runner"
echo "=============================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${BLUE}[1/6] Checking prerequisites...${NC}"

# Check API key
if [ -z "$AZURE_OPENAI_API_KEY" ]; then
    echo -e "${RED}❌ Error: AZURE_OPENAI_API_KEY environment variable not set${NC}"
    echo "Please run: export AZURE_OPENAI_API_KEY=\"your-api-key-here\""
    exit 1
fi
echo -e "${GREEN}✓ Azure OpenAI API key found${NC}"

# Check dataset exists
if [ ! -f "$PROJECT_ROOT/data/swebench_rebench/full_with_images.jsonl" ]; then
    echo -e "${RED}❌ Error: Full dataset with Docker images not found${NC}"
    echo "Please run: python scripts/filter_swebench_full_with_images.py"
    exit 1
fi

# Count instances
INSTANCE_COUNT=$(wc -l < "$PROJECT_ROOT/data/swebench_rebench/full_with_images.jsonl")
echo -e "${GREEN}✓ Found $INSTANCE_COUNT instances with Docker images${NC}"

# Check Docker
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}❌ Error: Docker is not running${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker is running${NC}"
echo ""

# Navigate to project root
cd "$PROJECT_ROOT"

# Stop any existing containers
echo -e "${BLUE}[2/6] Stopping existing containers...${NC}"
cd extra
docker compose down 2>/dev/null || true
echo -e "${GREEN}✓ Existing containers stopped${NC}"
echo ""

# Build the container
echo -e "${BLUE}[3/6] Building SWE-bench container (this may take a few minutes)...${NC}"
docker compose build swebench-rebench-full-with-images
echo -e "${GREEN}✓ Container built${NC}"
echo ""

# Start services
echo -e "${BLUE}[4/6] Starting services...${NC}"
docker compose up -d controller redis swebench-rebench-full-with-images

# Wait for services
echo -e "${BLUE}[5/6] Waiting for services to be ready...${NC}"
sleep 15

# Verify services
echo "Checking service status..."
CONTROLLER_STATUS=$(curl -s http://localhost:5020/api/status 2>/dev/null || echo "failed")
WORKER_STATUS=$(curl -s http://localhost:5032/api/status 2>/dev/null || echo "failed")

if [[ "$CONTROLLER_STATUS" == "failed" ]]; then
    echo -e "${RED}❌ Controller not responding${NC}"
    echo "Checking logs:"
    docker compose logs --tail=20 controller
    exit 1
fi

if [[ "$WORKER_STATUS" == "failed" ]]; then
    echo -e "${RED}❌ SWE-bench worker not responding${NC}"
    echo "Checking logs:"
    docker compose logs --tail=20 swebench-rebench-full-with-images
    exit 1
fi

echo -e "${GREEN}✓ All services ready${NC}"
echo ""

# Run the benchmark
echo -e "${BLUE}[6/6] Starting SWE-bench Full Dataset with Docker Images...${NC}"
echo ""
echo -e "${YELLOW}⚡ Running $INSTANCE_COUNT instances${NC}"
echo -e "${YELLOW}📊 This is the COMPLETE SWE-bench dataset filtered for Docker images${NC}"
echo -e "${YELLOW}⏱️  Expected duration: 8-24 hours depending on rate limits${NC}"
echo -e "${YELLOW}📁 Results will be saved to: outputs/[timestamp]/swebench_rebench_full_with_images_predictions.jsonl${NC}"
echo ""
echo "Starting benchmark in 5 seconds..."
sleep 1
echo "4..."
sleep 1
echo "3..."
sleep 1
echo "2..."
sleep 1
echo "1..."
sleep 1
echo ""

cd "$PROJECT_ROOT"
python -m src.assigner --config configs/assignments/swebench_rebench_full_with_images.yaml

echo ""
echo "=============================================================="
echo -e "${GREEN}🎉 SWE-bench Full Dataset with Docker Images Complete!${NC}"
echo "=============================================================="
echo ""
echo "Results location:"
OUTPUT_DIR=$(ls -td outputs/*/ 2>/dev/null | head -n1)
if [ -n "$OUTPUT_DIR" ]; then
    echo "  $OUTPUT_DIR"
    if [ -f "${OUTPUT_DIR}swebench_rebench_full_with_images_predictions.jsonl" ]; then
        RESULTS_COUNT=$(wc -l < "${OUTPUT_DIR}swebench_rebench_full_with_images_predictions.jsonl")
        echo "  Generated $RESULTS_COUNT predictions"
    fi
fi
echo ""
echo "To evaluate the results with the official SWE-bench harness:"
echo "  ./scripts/evaluate_patches.sh $OUTPUT_DIR/swebench_rebench_full_with_images_predictions.jsonl"
echo ""