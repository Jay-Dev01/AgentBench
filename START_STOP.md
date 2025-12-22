# AgentBench Start/Stop Guide

Quick reference for starting and stopping AgentBench on macOS.

---

## Stop Everything

### Quick Stop (All at Once)
```bash
# Kill the benchmark process and stop Docker containers
pkill -f "python3 -m src.assigner" && \
cd /Users/anubhav/Documents/FALL25/ALGOVERSE/agentbenchmod/AgentBench-feat-uncertainty-quantifier/extra && \
docker compose down
```

### Step-by-Step Stop

**1. Stop the benchmark process:**
```bash
pkill -f "python3 -m src.assigner"
```

**2. Stop Docker containers:**
```bash
cd /Users/anubhav/Documents/FALL25/ALGOVERSE/agentbenchmod/AgentBench-feat-uncertainty-quantifier/extra
docker compose down
```

---

## Start Everything

### Option 1: Using the Run Script (Recommended)

```bash
# Navigate to project directory
cd /Users/anubhav/Documents/FALL25/ALGOVERSE/agentbenchmod/AgentBench-feat-uncertainty-quantifier

# Set your Azure OpenAI API key
export AZURE_OPENAI_API_KEY="your-api-key-here"

# Run the task setup script
./run_task_mac.sh alfworld-std

# After services are ready (the script will tell you), run the benchmark:
python3 -m src.assigner
```

### Option 2: Manual Start

**1. Start Docker services:**
```bash
cd /Users/anubhav/Documents/FALL25/ALGOVERSE/agentbenchmod/AgentBench-feat-uncertainty-quantifier/extra
docker compose up -d controller redis alfworld-std
```

**2. Wait for services to initialize:**
```bash
sleep 15
```

**3. Verify worker registration:**
```bash
curl -s http://localhost:5020/api/list_workers | python3 -m json.tool
```

**4. Run the benchmark:**
```bash
cd /Users/anubhav/Documents/FALL25/ALGOVERSE/agentbenchmod/AgentBench-feat-uncertainty-quantifier

# Set your API key
export AZURE_OPENAI_API_KEY="your-api-key-here"

# Run the assigner
python3 -m src.assigner
```

---

## Check Status

### Check if Docker containers are running:
```bash
docker ps
```

Expected output should show:
- `agentrl-controller`
- `redis`
- `agentbench-fc-alfworld-std-1`

### Check if benchmark is running:
```bash
ps aux | grep "python3 -m src.assigner" | grep -v grep
```

### Check worker status:
```bash
curl -s http://localhost:5020/api/list_workers | python3 -m json.tool
```

### Monitor benchmark progress:
```bash
# Watch the output files being created
ls -lh outputs/gpt-4o-mini/alfworld-std/

# Tail the latest results
tail -f outputs/gpt-4o-mini/alfworld-std/*.jsonl
```

---

## Running Different Tasks

To switch to a different task (e.g., dbbench, os-std, kg-std, webshop-std):

```bash
# Stop current containers
cd extra
docker compose down

# Start different task
cd ..
./run_task_mac.sh dbbench-std  # or os-std, kg-std, webshop-std

# Run the benchmark
python3 -m src.assigner
```

---

## Troubleshooting

### If services won't start:
```bash
# Check Docker is running
docker info

# Rebuild containers if needed
cd extra
docker compose build alfworld-std
docker compose up -d controller redis alfworld-std
```

### If benchmark fails with connection errors:
```bash
# Verify worker is accessible
curl http://localhost:5021/api/get_sessions

# Check controller is running
curl http://localhost:5020/api/list_workers
```

### If you get API rate limit errors:
The benchmark will automatically retry. You can:
1. Wait for the rate limit to reset
2. Increase your Azure OpenAI quota at https://aka.ms/oai/quotaincrease

---

## Notes

- **API Key**: Replace `"your-api-key-here"` with your actual Azure OpenAI API key
- **Duration**: Running all 109 alfworld tasks can take 1-2 hours depending on rate limits
- **Results**: Output files are saved in `outputs/gpt-4o-mini/alfworld-std/`
- **Persistence**: Stopping and restarting will resume from where you left off
