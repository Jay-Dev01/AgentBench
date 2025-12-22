# SWE-rebench Integration - Setup Guide

## Overview

SWE-rebench has been successfully integrated into your AgentBench project! This new task allows agents to explore code in pre-built Docker containers and generate patches to fix real-world software engineering bugs.

## What Was Created

### Task Implementation (`src/server/tasks/swebench_rebench/`)

- ✅ `task.py` - Main task class with interactive agent loop
- ✅ `environment.py` - Docker environment delegation (4GB RAM, 2 vCPUs per container)
- ✅ `container.py` - Container management utilities
- ✅ `__init__.py` - Module exports
- ✅ `requirements.txt` - Python dependencies (datasets, docker)
- ✅ `Dockerfile` - Worker container configuration

### Configuration Files

- ✅ `configs/tasks/swebench_rebench.yaml` - Task configuration with 4 tools
- ✅ `configs/assignments/swebench_rebench_test.yaml` - Assignment configuration
- ✅ Updated `configs/tasks/task_assembly.yaml` - Added import for new task
- ✅ Updated `extra/docker-compose.yml` - Added swebench-rebench-dev service (port 5028)

### Scripts

- ✅ `scripts/download_swebench_rebench.py` - Dataset downloader
- ✅ `scripts/evaluate_swebench_rebench.sh` - Evaluation script

## Quick Start

### Step 0: Configure Direct Worker Communication (REQUIRED)

**IMPORTANT**: Before running SWE-bench tasks, you must configure direct worker communication to avoid "INTERACT_FAILED" errors.

Edit `src/client/task.py` and ensure the `WORKER_ADDRESSES` mapping includes:

```python
WORKER_ADDRESSES = {
    "alfworld-std": "http://localhost:5021/api",
    "dbbench-std": "http://localhost:5022/api",
    "os-std": "http://localhost:5023/api",
    "kg-std": "http://localhost:5024/api",
    "webshop-std": "http://localhost:5025/api",
    "swebench-rebench-dev": "http://localhost:5028/api",  # This line is critical!
}
```

### Step 1: Download the Dataset

**REQUIRED**: Dataset files are not included in the repository. Run this script to download them:

```bash
# Install the datasets library if not already installed
pip install datasets

# Download and prepare the SWE-rebench dataset (~900MB total)
python scripts/download_swebench_rebench.py
```

This will create:

- `data/swebench_rebench/dev.jsonl` (100 instances)
- `data/swebench_rebench/standard.jsonl` (1,000 instances)
- `data/swebench_rebench/full.jsonl` (21,336 instances)

**Note**: These large data files are gitignored and will not be committed to the repository.

### Step 1b: Filter Dataset to Include Only Samples with Docker Images (Recommended)

Not all SWE-bench instances have pre-built Docker images available. To ensure reliable execution, filter the datasets to only include instances with Docker images:

```bash
# Filter dev dataset (creates dev_with_images.jsonl)
python scripts/filter_swebench_dev_with_images.py

# Filter standard dataset (creates standard_with_images.jsonl)
python scripts/filter_swebench_standard_with_images.py

# Create a smaller 100-sample test set (creates standard_100_with_images.jsonl)
python scripts/filter_swebench_standard_100_with_images.py

# Filter full dataset (creates full_with_images.jsonl) - this may take a few minutes
python scripts/filter_swebench_full_with_images.py
```

**Filtered datasets created:**

- `dev_with_images.jsonl` - Dev set with only instances that have Docker images
- `standard_100_with_images.jsonl` - First 100 samples from standard set with Docker images (great for quick testing!)
- `standard_with_images.jsonl` - Full standard set with only instances that have Docker images
- `full_with_images.jsonl` - Full dataset with only instances that have Docker images

**Why filter?** Using filtered datasets prevents failures from missing Docker images and ensures faster, more reliable execution. The `standard_100_with_images.jsonl` is particularly useful for quick validation and testing.

### Step 2: Configure Your Agent

Make sure you have an agent configured. The test configuration uses `gpt-4o-mini`. Update your agent config in:

- `configs/agents/` (e.g., `gpt4-chat.yaml` or your custom agent config)

Example agent configuration should include your API credentials with the correct API version:

```yaml
# configs/agents/openai-chat.yaml
module: src.client.agents.HTTPAgent
parameters:
  url: https://your-resource.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-10-01-preview
  headers:
    Content-Type: application/json
    api-key: ${AZURE_OPENAI_API_KEY}
  body:
    temperature: 0
  prompter:
    name: openai_passthrough
  return_format: openai_chat
```

**Note**: The API version `2024-10-01-preview` is required for proper function calling support.

### Step 3: Run the Task

#### Option A: Using Docker Compose (Recommended)

```bash
# Navigate to the extra directory
cd extra

# Build the swebench-rebench-dev container or whatever task u want
docker-compose build swebench-rebench-dev


# Start all required services
docker-compose up -d controller redis swebench-rebench-dev

# Verify all containers are running
docker ps
# You should see 3 containers: controller, redis, swebench-rebench-dev

# Check SWE-bench logs for any startup issues
docker-compose logs swebench-rebench-dev

# Test connectivity (should return JSON with worker info)
curl http://localhost:5028/api/status 2>/dev/null || echo "SWE-bench worker not ready yet"
```

**Wait for all services to be ready**:

```bash
# Check that all containers are running
docker ps

# You should see:
# - agentrl-controller (port 5020)
# - redis (port 6379)
# - agentbench-fc-swebench-rebench-dev-1 (port 5028)
```

Then run the assignment:

```bash
# From project root
# Make sure AZURE_OPENAI_API_KEY is set
export AZURE_OPENAI_API_KEY="your-api-key-here"

python -m src.assigner --config configs/assignments/swebench_rebench_test.yaml
```

#### Option B: Direct Execution (Without Docker Compose)

```bash
# From project root
python -m src.assigner --config configs/assignments/swebench_rebench_test.yaml
```

### Step 4: Monitor Progress

Predictions will be saved to:

```
outputs/{timestamp}/swebench_rebench_dev_predictions.jsonl
```

Each line contains:

```json
{
  "instance_id": "repo-name-issue-id",
  "model_patch": "diff content...",
  "model_name_or_path": "agentbench"
}
```

### Step 5: Evaluate Results (Optional)

To evaluate the generated patches with the official SWE-bench harness:

```bash
# Install SWE-bench harness (if not already installed)
# git clone https://github.com/SWE-rebench/SWE-bench-fork.git
# cd SWE-bench-fork && pip install -e .

# Run evaluation
./scripts/evaluate_swebench_rebench.sh outputs/{timestamp}/swebench_rebench_dev_predictions.jsonl
```

This will:

1. Apply patches in fresh Docker containers
2. Run test suites (checking FAIL_TO_PASS tests)
3. Generate a report in `logs/run_evaluation/{run-id}/`

## Agent Tools

The agent has access to 4 tools for interactive code exploration:

### 1. `bash_command(command)`

Execute bash commands in the container at /testbed.

**Examples:**

```python
bash_command("ls -la")
bash_command("git log -1")
bash_command("find . -name '*.py' | head -20")
bash_command("grep -r 'def myfunction' src/")
bash_command("pytest tests/test_utils.py -v")
```

### 2. `read_file(file_path)`

Read file contents.

**Examples:**

```python
read_file("src/main.py")
read_file("tests/test_utils.py")
read_file("README.md")
```

### 3. `search_code(pattern, directory='.')`

Search for patterns in Python files.

**Examples:**

```python
search_code("class MyClass")
search_code("def process_data", "src/")
search_code("import numpy")
```

### 4. `submit_patch(patch)`

Submit the final patch (ends the task).

**Example:**

```python
submit_patch("""diff --git a/src/main.py b/src/main.py
index 123abc..456def 100644
--- a/src/main.py
+++ b/src/main.py
@@ -10,7 +10,7 @@ def process():
-    return None
+    return result
""")
```

## Configuration Options

### Task Parameters (`configs/tasks/swebench_rebench.yaml`)

```yaml
max_round: 20 # Maximum interaction rounds per instance
command_timeout: 60 # Timeout for bash commands (seconds)
concurrency: 8 # Max concurrent containers
```

### Assignment Parameters (`configs/assignments/swebench_rebench_test.yaml`)

```yaml
concurrency:
  task:
    swebench-rebench-dev: 1 # Number of instances to run in parallel
  agent:
    gpt-4o-mini: 1 # Agent concurrency
```

### Docker Image Management

The task uses pre-built Docker images from Docker Hub (`swerebench/{instance_id}:latest`). These images:

- Already contain the repository at `/testbed`
- Have all dependencies installed
- Are ready to use (no build time needed)

## Troubleshooting

### "Dataset not found"

```bash
# Make sure you downloaded the dataset
python scripts/download_swebench_rebench.py
```

### "Docker image pull failed"

```bash
# Check Docker is running
docker ps

# Check internet connection
# The images are pulled from Docker Hub on demand
```

### "Container creation timeout" or "NETWORK_ERROR Read timed out"

**Problem**: Tasks timeout during Docker image pulling or complex operations.

**Solutions**:

1. Increase timeout in `configs/tasks/swebench_rebench.yaml`:

```yaml
parameters:
  command_timeout: 120 # Increase from 60 seconds
```

2. Pre-pull commonly used Docker images:

```bash
# Pre-pull some SWE-bench evaluation images
docker exec agentbench-fc-swebench-rebench-dev-1 docker pull swerebench/sweb.eval.x86_64.tobymao_1776_sqlglot-6385
```

3. Reduce concurrency if system is under resource pressure:

```yaml
# In configs/assignments/swebench_rebench_test.yaml
concurrency:
  task:
    swebench-rebench-dev: 1 # Start with 1, increase as needed
```

### "INTERACT_FAILED" Errors

**Problem**: Agents fail with "failed to interact with session" errors.

**Solution**: Ensure direct worker communication is configured:

1. Check that `src/client/task.py` has `"swebench-rebench-dev": "http://localhost:5028/api"` in `WORKER_ADDRESSES`
2. Verify the SWE-bench container is running on port 5028: `docker ps | grep swebench`
3. Check the API version in your agent config is `2024-10-01-preview` or newer

### "Agent not making tool calls"

**Problem**: Agent generates text responses instead of function calls.

**Solution**:

1. Update your OpenAI agent configuration to use API version `2024-10-01-preview`
2. Ensure the agent configuration includes proper `prompter` and `return_format` settings
3. Verify your Azure OpenAI deployment supports function calling

## Performance Tips

### 1. Run Multiple Instances in Parallel

```yaml
# In configs/assignments/swebench_rebench_test.yaml
concurrency:
  task:
    swebench-rebench-dev: 4 # Run 4 instances at once
```

### 2. Use Smaller Dataset for Testing

Start with the dev set (100 instances) before running the full dataset.

### 3. Monitor Resource Usage

Each container uses up to 4GB RAM and 2 vCPUs. Ensure your system has sufficient resources:

- For 4 parallel instances: ~16GB RAM + 8 vCPUs

## Dataset Variants

Switch between datasets by changing the assignment config:

```yaml
# configs/assignments/swebench_rebench_test.yaml
assignments:
  - agent: [gpt-4o-mini]
    task:
      - swebench-rebench-dev # 100 instances
      # - swebench-rebench-std    # 1,000 instances
      # - swebench-rebench-full   # 21,336 instances
```

## Expected Results

For baseline LLM agents (without retrieval or advanced techniques):

- **Pass rate**: 5-15% on dev set
- **Patch generation rate**: 70-90% (agents attempt to submit patches)
- **Completion rate**: 85-95% (tasks complete without errors)

Better agents with retrieval, test feedback, or iterative refinement can achieve higher pass rates.

## Next Steps

1. **Test with 1 instance**: Modify dev.jsonl to have just 1 line and test end-to-end
2. **Run dev set**: Run all 100 instances in the dev set
3. **Analyze failures**: Look at instances that timeout or error out
4. **Iterate on prompts**: Improve system prompt in `task.py` if needed
5. **Scale up**: Try standard set (1,000) or full set (21,336)
6. **Evaluate**: Use official harness to get actual pass rates

## Architecture

```
Agent (GPT-4, etc)
    ↓ uses tools
Docker Container (swerebench/instance:latest)
    ↓ contains
Repository at /testbed
    ↓ agent explores and fixes
Generated Patch
    ↓ saved to
predictions.jsonl
    ↓ evaluated by
Official SWE-bench Harness
```

## Support

If you encounter issues:

1. Check the logs: `docker-compose logs swebench-rebench-dev`
2. Verify dataset exists: `ls data/swebench_rebench/`
3. Check Docker: `docker ps` and `docker images`
4. Review task config: `configs/tasks/swebench_rebench.yaml`

## Summary

You now have a fully functional SWE-rebench task integrated into AgentBench! The implementation follows AgentBench patterns, uses pre-built Docker images for speed, and provides interactive code exploration for agents.

Happy bug hunting! 🐛🔧
