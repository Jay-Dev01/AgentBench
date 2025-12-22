# SWE-bench Integration for AgentBench

A comprehensive guide to set up and run SWE-bench tasks in AgentBench from scratch.

## Overview

SWE-bench is a benchmark for evaluating AI agents on real-world software engineering tasks. This integration allows agents to explore code repositories in Docker containers and generate patches to fix bugs.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.10+
- Azure OpenAI API access
- At least 8GB RAM and 4 CPU cores for parallel execution

## Step-by-Step Setup

### 1. Download the SWE-bench Dataset

```bash
# Install required Python packages
pip install datasets

# Download the SWE-bench rebench dataset
python scripts/download_swebench_rebench.py
```

This creates:

- `data/swebench_rebench/dev.jsonl` (100 instances - full dev set)
- `data/swebench_rebench/dev_with_images.jsonl` (34 instances - only with Docker images)
- `data/swebench_rebench/standard.jsonl` (1,000 instances)
- `data/swebench_rebench/full.jsonl` (21,336 instances)

### 2. Configure Direct Worker Communication (CRITICAL)

**This step is essential to avoid "INTERACT_FAILED" errors.**

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

### 3. Configure Azure OpenAI API

Update your agent configuration with the correct API version:

```bash
# Edit configs/agents/openai-chat.yaml
```

Ensure it contains:

```yaml
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

**Note**: The API version `2024-10-01-preview` is required for proper function calling.

### 4. Set Your API Key

```bash
export AZURE_OPENAI_API_KEY="your-azure-openai-api-key-here"
```

### 5. Choose Your Dataset

By default, the system uses `dev_with_images.jsonl` (34 instances with pre-built Docker images).

**To run the full dev set (100 instances):**

Edit `configs/tasks/swebench_rebench.yaml` and change:

```yaml
swebench-rebench-dev:
  parameters:
    name: "swebench-rebench-dev"
    data_file: "data/swebench_rebench/dev.jsonl" # Changed from dev_with_images.jsonl
    output_file: "outputs/swebench_rebench_dev_predictions.jsonl"
```

### 6. Build and Start Docker Services

```bash
# Navigate to the Docker compose directory
cd extra

# Build the SWE-bench container (required after config changes)
# or whatever dataset you wanna use, example swebench-rebench-100, swebench-rebench-full-with-images:

docker-compose build swebench-rebench-dev

# Start all required services
docker-compose up -d controller redis swebench-rebench-dev

# Verify all containers are running
docker ps
```

You should see:

- `agentrl-controller` on port 5020
- `redis` on port 6379
- `agentbench-fc-swebench-rebench-dev-1` on port 5028

### 7. Verify the Setup

```bash
# Check SWE-bench container logs
docker-compose logs swebench-rebench-dev

# Look for this line to confirm correct dataset loading:
# [INFO] [task.py:105]: Loaded 100 instances from SWE-rebench (for full dev set)
# or
# [INFO] [task.py:105]: Loaded 34 instances from SWE-rebench (for images-only)

# Test connectivity
curl http://localhost:5028/api/status 2>/dev/null || echo "SWE-bench worker not ready yet"
```

### 8. Run SWE-bench Tasks

```bash
# Navigate back to project root
cd ..

# Run the assignment
python -m src.assigner --config configs/assignments/swebench_rebench_test.yaml
```

## Understanding the Results

### Output Structure

Results are saved in timestamped directories:

```
outputs/YYYY-MM-DD-HH-MM-SS/gpt-4o-mini/swebench-rebench-dev/
├── runs.jsonl          # Detailed execution logs
├── error.jsonl         # Failed instances (if any)
└── predictions.jsonl   # Generated patches (if configured)
```

### Dataset Behavior

**Full dev set (100 instances):**

- 34 instances with Docker images → Agent interaction with pre-built environments
- 66 instances without images → Automatically skipped with empty patches (default behavior)

**To handle instances without pre-built images:**
You would need to modify `src/server/tasks/swebench_rebench/task.py` to build environments dynamically:

1. **Create base Docker images** from repository requirements
2. **Install dependencies** using pip/conda/npm based on repo config
3. **Set up the repository** at `/testbed` with proper git state
4. **Handle build failures** and dependency conflicts

This requires significant additional implementation work and would slow down execution considerably (2-10 minutes per instance for environment setup).

**Images-only set (34 instances):**

- All instances have Docker images → All get agent interaction

### Performance Monitoring

```bash
# Monitor progress during execution
watch -n 5 'docker-compose logs --tail=10 swebench-rebench-dev'

# Check container resource usage
docker stats
```

## Troubleshooting

### "INTERACT_FAILED" Errors

**Cause**: Missing direct worker address mapping.
**Solution**: Ensure Step 2 is completed - add `"swebench-rebench-dev": "http://localhost:5028/api"` to `WORKER_ADDRESSES`.

### "Authentication Failed" (401 errors)

**Cause**: Missing or invalid API key.
**Solution**:

```bash
export AZURE_OPENAI_API_KEY="your-key-here"
# Verify it's set
echo $AZURE_OPENAI_API_KEY
```

### Container Conflicts

**Cause**: Old configuration cached in containers.
**Solution**:

```bash
cd extra
docker-compose stop swebench-rebench-dev
docker-compose build swebench-rebench-dev
docker-compose restart controller
docker-compose up -d swebench-rebench-dev
```

### Network Timeouts

**Cause**: Docker image pulling takes too long.
**Solutions**:

1. Increase timeout in `configs/tasks/swebench_rebench.yaml`:

```yaml
parameters:
  command_timeout: 120 # Increase from 60
```

2. Pre-pull common images:

```bash
docker exec agentbench-fc-swebench-rebench-dev-1 docker pull swerebench/sweb.eval.x86_64.tobymao_1776_sqlglot-6385
```

### "Task already exists" Error

**Solution**: Restart the controller to clear state:

```bash
cd extra
docker-compose restart controller
docker-compose restart swebench-rebench-dev
```

## Configuration Options

### Concurrency Settings

Edit `configs/assignments/swebench_rebench_test.yaml`:

```yaml
concurrency:
  task:
    swebench-rebench-dev: 1 # Number of parallel instances
  agent:
    gpt-4o-mini: 1 # Agent concurrency
```

### Task Parameters

Edit `configs/tasks/swebench_rebench.yaml`:

```yaml
parameters:
  max_round: 20 # Max conversation rounds per task
  command_timeout: 60 # Timeout for bash commands
  concurrency: 8 # Max concurrent containers
```

### Different Datasets

Switch datasets by changing `data_file` in `configs/tasks/swebench_rebench.yaml`:

- `dev.jsonl` - Full dev set (100 instances)
- `dev_with_images.jsonl` - Only with Docker images (34 instances)
- `standard.jsonl` - Standard set (1,000 instances)
- `full.jsonl` - Full dataset (21,336 instances)

## Agent Tools

The agent has access to 4 tools for code exploration:

1. **`bash_command(command)`** - Execute bash commands in `/testbed`
2. **`read_file(file_path)`** - Read file contents
3. **`search_code(pattern, directory='.')`** - Search for patterns in Python files
4. **`submit_patch(patch)`** - Submit final unified diff patch

## Expected Performance

- **Completion time**: 15-60 seconds per instance with Docker image
- **Success rate**: 70-90% patch generation rate for baseline agents
- **Pass rate**: 5-15% for simple agents without advanced techniques

## Stopping the System

```bash
cd extra

# Stop specific services
docker-compose stop swebench-rebench-dev alfworld-std

# Stop all services
docker-compose down

# Stop and remove containers + images (full cleanup)
docker-compose down --rmi all
```

## Advanced Usage

### Handling Instances Without Pre-built Images

By default, instances without Docker images are skipped. To handle them, you would need to implement dynamic environment building:

**Required modifications in `src/server/tasks/swebench_rebench/task.py`:**

```python
# Instead of skipping, build environment dynamically
if not docker_image or docker_image == "None":
    # Create custom Docker image for this instance
    docker_image = await self._build_custom_environment(instance)

async def _build_custom_environment(self, instance):
    """Build a Docker environment from repository requirements."""
    repo = instance['repo']
    base_commit = instance['base_commit']

    # 1. Clone repository
    # 2. Parse requirements.txt, setup.py, pyproject.toml, etc.
    # 3. Build Docker image with dependencies
    # 4. Return image name

    # This could take 2-10 minutes per instance
    pass
```

**Challenges:**

- Complex dependency resolution
- Build failures and conflicts
- Significantly increased execution time
- Resource-intensive Docker builds
- Repository-specific setup requirements

**Alternative approach:** Use a generic Python/Node.js environment and install dependencies at runtime within the container.

### Custom Agent Configuration

Create a new agent config in `configs/agents/`:

```yaml
my-custom-agent:
  import: "./openai-chat.yaml"
  parameters:
    name: "my-custom-agent"
    body:
      temperature: 0.7
      max_tokens: 4096
```

### Running Multiple Datasets

Create separate assignment files for different datasets:

```bash
cp configs/assignments/swebench_rebench_test.yaml configs/assignments/swebench_full.yaml
# Edit to use swebench-rebench-full task
```

### Evaluation with Official Harness

```bash
# Install SWE-bench evaluation harness
git clone https://github.com/princeton-nlp/SWE-bench.git
cd SWE-bench && pip install -e .

# Run evaluation on generated predictions
python -m swebench.harness.run_evaluation \
  --predictions_path outputs/TIMESTAMP/swebench_rebench_dev_predictions.jsonl \
  --swe_bench_tasks princeton-nlp/SWE-bench_Lite \
  --log_dir logs/
```

## Architecture Overview

```
Agent (GPT-4o-mini)
    ↓ OpenAI Function Calls
AgentBench Task Client (port 5028)
    ↓ Direct Worker API
SWE-bench Container
    ↓ Docker-in-Docker
Evaluation Container (swerebench/*)
    ↓ Repository at /testbed
Generated Patch
    ↓ Saved to outputs/
Predictions JSONL
```

## Support

If you encounter issues:

1. **Check container logs**: `docker-compose logs swebench-rebench-dev`
2. **Verify dataset**: `ls -la data/swebench_rebench/`
3. **Test API key**: `echo $AZURE_OPENAI_API_KEY`
4. **Check ports**: `docker ps` and `netstat -tulpn | grep :502`
5. **Rebuild containers**: Follow troubleshooting steps above

## Summary

This integration provides a complete SWE-bench evaluation environment within AgentBench, supporting:

- ✅ Interactive code exploration with 4 tools
- ✅ Pre-built Docker evaluation environments
- ✅ Automatic patch generation and submission
- ✅ Full dataset coverage (34-21,336 instances)
- ✅ Scalable parallel execution
- ✅ Integration with Azure OpenAI function calling

Happy bug hunting! 🐛🔧
