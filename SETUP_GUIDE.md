# AgentBench Setup Guide (Azure OpenAI + WSL2)

This guide walks you through setting up and running AgentBench with Azure OpenAI on Windows using WSL2.

## Prerequisites
- Use Ubuntu
- Windows 10/11 with WSL2 enabled
- Docker Desktop for Windows with WSL2 integration enabled
- An Azure OpenAI resource with a deployed model (e.g., `gpt-4o-mini`)

---

## Step 1: Enable WSL2 Integration in Docker Desktop

1. Open **Docker Desktop**
2. Go to **Settings** → **Resources** → **WSL Integration**
3. Enable integration with your WSL2 distro (e.g., Ubuntu)
4. Click **Apply & Restart**

---

## Step 2: Clone the Repository

```bash
# In WSL2 terminal
cd ~
git clone https://github.com/Jay-Dev01/AgentBench.git
cd AgentBench
git checkout ubuntu-azure-setup
```

---

## Step 3: Set Up Python Environment

```bash
# Install Python 3.11 if not available
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 4: Configure Azure OpenAI API Key

Set your Azure OpenAI API key as an environment variable:

```bash
export AZURE_OPENAI_API_KEY="your-azure-api-key-here"
```

To make it persistent, add it to your `~/.bashrc`:

```bash
echo 'export AZURE_OPENAI_API_KEY="your-azure-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### Finding Your Azure OpenAI Credentials

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to your **Azure OpenAI resource**
3. Click **Keys and Endpoint**
4. Copy **Key 1** or **Key 2**

The configuration file (`configs/agents/openai-chat.yaml`) is already set up to use:
- **Endpoint**: `https://algoverse-ab.openai.azure.com/`
- **Deployment**: `gpt-4o-mini`
- **API Version**: `2024-08-01-preview`

If your Azure resource is different, update the URL in `configs/agents/openai-chat.yaml`.

---

## Step 5: Start Docker Services

```bash
cd ~/AgentBench/extra

# Start the controller, redis, and alfworld worker
docker compose up -d controller redis alfworld-std

# Wait for services to initialize (~30-60 seconds)

# Verify services are running
docker compose ps

# Check that the worker registered
curl http://localhost:5020/api/list_workers
```

You should see output showing `alfworld-std` with workers registered.

### Verify Direct Worker Access

```bash
curl http://localhost:5021/api/get_sessions
```

This should return `[]` or a list of sessions.

---

## Step 6: Run the Benchmark

```bash
cd ~/AgentBench
source venv/bin/activate

# Make sure API key is set
echo $AZURE_OPENAI_API_KEY

# Run the assigner
python -m src.assigner
```

### Expected Output

```
TaskClient created: alfworld-std (http://localhost:5020/api)
  -> Using direct worker address: http://localhost:5021/api
Message: 109 samples remaining.
Agent "gpt-4o-mini" needs to run 1 tasks with total 109 samples:
    Task "alfworld-std": 109
Running Count: 0
Assigned gpt-4o-mini/alfworld-std#108
...
```

The benchmark will run through 109 ALFWorld tasks. Results are saved to the `outputs/` directory.

---

## Troubleshooting

### Rate Limit Errors

If you see `RateLimitReached` errors, the concurrency is set to 1 in `configs/assignments/default.yaml` to minimize this. You can:

1. Wait and retry (the error message tells you how long)
2. Increase your Azure quota at [aka.ms/oai/quotaincrease](https://aka.ms/oai/quotaincrease)

### Connection Refused

If you get connection errors:

```bash
# Check Docker services are running
docker compose ps

# Check controller logs
docker logs agentrl-controller --tail 50

# Check worker logs
docker logs agentbench-fc-alfworld-std-1 --tail 50

# Restart services
docker compose down
docker compose up -d controller redis alfworld-std
```

### Worker Not Registering

If `curl http://localhost:5020/api/list_workers` shows empty workers:

```bash
# Check worker logs for errors
docker logs agentbench-fc-alfworld-std-1 --tail 100

# Rebuild and restart
docker compose down
docker compose build alfworld-std
docker compose up -d controller redis alfworld-std
```

---

## Configuration Files

| File | Purpose |
|------|---------|
| `configs/agents/openai-chat.yaml` | Azure OpenAI endpoint and API key |
| `configs/agents/api_agents.yaml` | Agent definitions (gpt-4o-mini) |
| `configs/assignments/default.yaml` | Task assignments and concurrency |
| `configs/assignments/definition.yaml` | Controller address (port 5020) |
| `extra/docker-compose.yml` | Docker service definitions |

---

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Python        │     │   Controller     │     │   Task          │
│   Assigner      │────▶│   (port 5020)    │────▶│   Workers       │
│                 │     │                  │     │   (ports 5021+) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                                                 │
        │  (direct communication - bypasses controller)   │
        └─────────────────────────────────────────────────┘
        
        │
        ▼
┌─────────────────┐
│   Azure OpenAI  │
│   (gpt-4o-mini) │
└─────────────────┘
```

**Note:** The Python client talks directly to the worker because the controller has a bug that prevents proper `/interact` forwarding.

---

## Stopping Services

```bash
cd ~/AgentBench/extra
docker compose down
```

---

## Available Tasks

AgentBench includes multiple benchmark tasks across different domains:

### Core Tasks (Build from source)

| Task | Description | Docker Service | Host Port |
|------|-------------|----------------|-----------|
| **alfworld-std** | Household tasks (ALFWorld) | `alfworld-std` | 5021 |
| **dbbench-std** | Database benchmark | `dbbench-std` | 5022 |
| **os-std** | OS interaction tasks | `os_interaction-std` | 5023 |
| **kg-std** | Knowledge graph (KGQA) | `knowledgegraph-std` | 5024 |
| **webshop-std** | Web shopping tasks | `webshop-std` | 5025 |

### ToolEmu Tasks (Build from source)

| Task | Description | Docker Service | Host Port |
|------|-------------|----------------|-----------|
| **toolemu-std** | Standard tool emulation | `toolemu-std` | 5026 |
| **toolemu-adv** | Adversarial mode (30% failure) | `toolemu-adv` | 5027 |
| **toolemu-stress** | Stress mode (50% failure) | `toolemu-stress` | 5028 |
| **toolemu-safety** | Safety-focused evaluation | `toolemu-safety` | 5029 |

### Pre-built Image Tasks (Pull from Docker Hub)

| Task | Description | Docker Service | Host Port |
|------|-------------|----------------|-----------|
| **m2w-std** | Mind2Web standard | `mind2web-std` | 5030 |
| **m2w-dev** | Mind2Web dev set | `mind2web-dev` | 5031 |
| **cg-std** | Card Game (Aquawar) standard | `card_game-std` | 5032 |
| **cg-dev** | Card Game dev set | `card_game-dev` | 5033 |
| **ltp-std** | Lateral Thinking Puzzles std | `ltp-std` | 5034 |
| **ltp-dev** | Lateral Thinking Puzzles dev | `ltp-dev` | 5035 |
| **avalon-dev-naive** | Avalon naive mode | `avalon-dev-naive` | 5036 |
| **avalon-dev-single** | Avalon single mode | `avalon-dev-single` | 5037 |

---

## Running Different Tasks

### Option 1: Quick Start

#### 1. Edit `configs/assignments/default.yaml`

Uncomment the task you want to run:

```yaml
    task:
      # - alfworld-std        # House-holding tasks
      - dbbench-std           # Database tasks (uncomment this one)
      # - os-std              # OS interaction tasks
      # ... etc
```

#### 2. Start the Docker service

```bash
cd ~/AgentBench/extra

# For core tasks (build from source)
docker compose up -d controller redis <service-name>

# For pre-built tasks (pull from Docker Hub)
docker compose up -d controller redis <service-name>
```

#### 3. Run the assigner

```bash
cd ~/AgentBench
source venv/bin/activate
python -m src.assigner
```

### Option 2: Use the run script

```bash
chmod +x run_task.sh
./run_task.sh alfworld-std   # or any other task name
```

---

## Task-Specific Commands

### ALFWorld (alfworld-std)

```bash
docker compose up -d controller redis alfworld-std
```

### Database Benchmark (dbbench-std)

```bash
docker compose up -d controller redis dbbench-std
```

### OS Interaction (os-std)

Build the required Docker images first:

```bash
cd ~/AgentBench
docker build -t local-os/default -f data/os_interaction/res/dockerfiles/default data/os_interaction/res/dockerfiles
docker build -t local-os/packages -f data/os_interaction/res/dockerfiles/packages data/os_interaction/res/dockerfiles
docker build -t local-os/ubuntu -f data/os_interaction/res/dockerfiles/ubuntu data/os_interaction/res/dockerfiles
```

Then start:

```bash
docker compose up -d controller redis os_interaction-std
```

### Knowledge Graph (kg-std)

Requires Freebase data:

1. Download data from [Freebase-Setup](https://github.com/dki-lab/Freebase-Setup)
2. Extract and place at `./extra/virtuoso_db/virtuoso.db`
3. Start with:

```bash
docker compose up -d controller redis knowledgegraph-std freebase
```

### WebShop (webshop-std)

- Requires ~16GB RAM
- Takes ~3 minutes to start

```bash
docker compose up -d controller redis webshop-std
```

### ToolEmu Tasks

```bash
# Standard mode
docker compose up -d controller redis toolemu-std

# Adversarial mode (30% failure injection)
docker compose up -d controller redis toolemu-adv

# Stress mode (50% failure injection)
docker compose up -d controller redis toolemu-stress

# Safety-focused evaluation
docker compose up -d controller redis toolemu-safety
```

### Mind2Web (m2w-std, m2w-dev)

```bash
# Standard set
docker compose up -d controller redis mind2web-std

# Dev set
docker compose up -d controller redis mind2web-dev
```

### Card Game / Aquawar (cg-std, cg-dev)

```bash
# Standard set
docker compose up -d controller redis card_game-std

# Dev set
docker compose up -d controller redis card_game-dev
```

### Lateral Thinking Puzzles (ltp-std, ltp-dev)

```bash
# Standard set
docker compose up -d controller redis ltp-std

# Dev set
docker compose up -d controller redis ltp-dev
```

### Avalon (avalon-dev-naive, avalon-dev-single)

```bash
# Naive mode
docker compose up -d controller redis avalon-dev-naive

# Single mode
docker compose up -d controller redis avalon-dev-single
```

---

## Port Mapping Reference

| Task | Host Port | Internal Port |
|------|-----------|---------------|
| Controller | 5020 | 5020 |
| Redis | 6379 | 6379 |
| alfworld-std | 5021 | 5021 |
| dbbench-std | 5022 | 5021 |
| os-std | 5023 | 5021 |
| kg-std | 5024 | 5021 |
| webshop-std | 5025 | 5021 |
| toolemu-std | 5026 | 5021 |
| toolemu-adv | 5027 | 5021 |
| toolemu-stress | 5028 | 5021 |
| toolemu-safety | 5029 | 5021 |
| m2w-std | 5030 | 5021 |
| m2w-dev | 5031 | 5021 |
| cg-std | 5032 | 5021 |
| cg-dev | 5033 | 5021 |
| ltp-std | 5034 | 5021 |
| ltp-dev | 5035 | 5021 |
| avalon-dev-naive | 5036 | 5021 |
| avalon-dev-single | 5037 | 5021 |

---

## License

Apache-2.0 - See [LICENSE](LICENSE) for details.
