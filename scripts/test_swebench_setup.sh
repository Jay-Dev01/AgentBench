#!/bin/bash
# Test SWE-bench Docker setup

echo "=========================================="
echo "SWE-bench Docker Setup Verification"
echo "=========================================="
echo ""

ERRORS=0

# Check 1: Python modules
echo "[1/7] Checking Python installation..."
if python --version &>/dev/null; then
    echo "✓ Python found: $(python --version)"
else
    echo "✗ Python not found"
    ERRORS=$((ERRORS+1))
fi
echo ""

# Check 2: Docker
echo "[2/7] Checking Docker..."
if docker --version &>/dev/null; then
    echo "✓ Docker found: $(docker --version)"
    if docker ps &>/dev/null; then
        echo "✓ Docker is running"
    else
        echo "✗ Docker is not running or permission denied"
        ERRORS=$((ERRORS+1))
    fi
else
    echo "✗ Docker not found"
    ERRORS=$((ERRORS+1))
fi
echo ""

# Check 3: SWE-bench data
echo "[3/7] Checking SWE-bench data files..."
if [ -f "data/swebench/dev.jsonl" ]; then
    LINES=$(wc -l < data/swebench/dev.jsonl | tr -d ' ')
    echo "✓ dev.jsonl found ($LINES instances)"
else
    echo "✗ data/swebench/dev.jsonl not found"
    ERRORS=$((ERRORS+1))
fi
echo ""

# Check 4: Task implementation
echo "[4/7] Checking task implementation..."
if [ -f "src/server/tasks/swebench_docker/task.py" ]; then
    echo "✓ SWE-bench Docker task found"
else
    echo "✗ Task implementation not found"
    ERRORS=$((ERRORS+1))
fi
echo ""

# Check 5: Configuration files
echo "[5/7] Checking configuration files..."
CONFIG_OK=1
if [ -f "configs/tasks/swebench_docker.yaml" ]; then
    echo "✓ Task config found"
else
    echo "✗ Task config not found"
    CONFIG_OK=0
fi

if [ -f "configs/assignments/swebench-docker-test.yaml" ]; then
    echo "✓ Assignment config found"
else
    echo "✗ Assignment config not found"
    CONFIG_OK=0
fi

if [ $CONFIG_OK -eq 0 ]; then
    ERRORS=$((ERRORS+1))
fi
echo ""

# Check 6: API key
echo "[6/7] Checking API key..."
if [ -f ".env" ]; then
    if grep -q "AZURE_OPENAI_API_KEY" .env; then
        echo "✓ .env file found with API key"
    else
        echo "⚠ .env file found but no AZURE_OPENAI_API_KEY"
        echo "  Add: AZURE_OPENAI_API_KEY=\"your-key-here\""
    fi
else
    echo "⚠ .env file not found"
    echo "  Create it with: echo 'AZURE_OPENAI_API_KEY=\"your-key\"' > .env"
fi
echo ""

# Check 7: SWE-bench harness
echo "[7/7] Checking SWE-bench harness..."
if [ -d "swe-bench-reference" ]; then
    echo "✓ SWE-bench reference repository found"
    if python -c "import swebench" 2>/dev/null; then
        echo "✓ SWE-bench module installed"
    else
        echo "⚠ SWE-bench module not installed"
        echo "  Install with: cd swe-bench-reference && pip install -e ."
    fi
else
    echo "✗ swe-bench-reference not found"
    echo "  Clone it with: git clone https://github.com/SWE-rebench/SWE-bench-fork.git swe-bench-reference"
    ERRORS=$((ERRORS+1))
fi
echo ""

# Summary
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo "✓ All checks passed!"
    echo ""
    echo "You're ready to run SWE-bench tasks:"
    echo "  ./scripts/run_swebench_docker.sh"
else
    echo "✗ $ERRORS check(s) failed"
    echo ""
    echo "Please fix the errors above before running."
fi
echo "=========================================="
