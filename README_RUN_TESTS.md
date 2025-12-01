## Run the Full AgentBench Test Suite (Windows-friendly)

This guide shows exactly how to run AgentBench end-to-end on Windows (PowerShell), either:
- Locally for the quick OS + DB tasks, or
- With Docker Compose for all major tasks (including ToolEmu, ALFWorld, WebShop, KnowledgeGraph).

If you only need the uncertainty demo, see `TESTING_UNCERTAINTY.md`.

---

### 0) Prerequisites
- Docker Desktop (Linux containers) and WSL2 enabled
- Python 3.9
- A Gemini API key (`GEMINI_API_KEY`)

Open Windows PowerShell:

```powershell
cd C:\Users\Jason\Documents\Algoverse-proj\AgentBench
```

Create and activate a virtual environment (choose one):

```powershell
# Option A: venv (built-in)
py -3.9 -m venv .venv
. .\.venv\Scripts\Activate.ps1

# Option B: conda
conda create -n agent-bench python=3.9 -y
conda activate agent-bench
```

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

Configure your agent credentials:
1) Set the environment variable for Gemini:
```powershell
$env:GEMINI_API_KEY = "<YOUR_GEMINI_API_KEY>"
```
Optionally specify model/API base (defaults work for most):
```powershell
$env:GEMINI_MODEL = "gemini-2.5-pro"   # or "gemini-2.5-flash"
$env:GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1"  # default
```
2) `configs/agents/api_agents.yaml` already includes `my-gemini` (which imports `configs/agents/my-gemini.yaml`). You typically don’t need to edit YAML if the env var is set.

Quickly verify the agent config (optional):

```powershell
python -m src.client.agent_test --config configs/agents/api_agents.yaml --agent my-gemini
```

---

## Path A: Quick Local Run (DB + OS)
Good for validating your setup quickly without heavy containers.

1) Ensure Docker works:
```powershell
docker --version
docker ps
```

2) Prepare required images:
```powershell
docker pull mysql:8
docker pull ubuntu
docker build -f data/os_interaction/res/dockerfiles/default data/os_interaction/res/dockerfiles --tag local-os/default
docker build -f data/os_interaction/res/dockerfiles/packages data/os_interaction/res/dockerfiles --tag local-os/packages
docker build -f data/os_interaction/res/dockerfiles/ubuntu data/os_interaction/res/dockerfiles --tag local-os/ubuntu
```

3) Start task services (controller + workers) for DB and OS:
```powershell
python -m src.start_task -a
```
- This spins up the controller on port 5000 and launches 5 workers each for `dbbench-std` and `os-std`.
- Wait ~1 minute until you see “200 OK” logs before proceeding.

4) Run the assignment (DB + OS):
```powershell
python -m src.assigner --config configs/assignments/default.yaml
```

5) View results:
- Outputs will be under `outputs\{TIMESTAMP}\{agent}\{task}\`
- Key files:
  - `runs.jsonl`: per-sample results
  - `error.jsonl`: failures
  - `overall.json`: aggregate metrics

---

## Path B: Full Stack with Docker Compose (All Major Tasks)
Starts a full stack including AgentRL controller and workers for:
- ToolEmu (std/adv/stress/safety)
- ALFWorld
- DBBench
- KnowledgeGraph (requires data)
- OS Interaction
- WebShop

1) Pre-build/pull images needed by workers that launch nested containers:
```powershell
docker pull mysql:8
docker build -t local-os/default -f ./data/os_interaction/res/dockerfiles/default data/os_interaction/res/dockerfiles
docker build -t local-os/packages -f ./data/os_interaction/res/dockerfiles/packages data/os_interaction/res/dockerfiles
docker build -t local-os/ubuntu -f ./data/os_interaction/res/dockerfiles/ubuntu data/os_interaction/res/dockerfiles
```

2) (Required for KnowledgeGraph) Download Freebase data
- Get data from `https://github.com/dki-lab/Freebase-Setup`
- Place the database at:
  - `AgentBench/extra/virtuoso_db/virtuoso.db`
  - If you place it elsewhere, update the volume in `extra/docker-compose.yml`.

3) Start the stack:
```powershell
docker compose -f extra/docker-compose.yml up -d
```
- This exposes the controller on `http://localhost:5020/api`
- Workers will register automatically as they come up

4) Run assignments against the full stack
The assignment configs already point at the compose controller (`configs/assignments/definition.yaml` sets `controller_address: "http://localhost:5020/api"`).

- ToolEmu (Gemini): create `configs/assignments/toolemu-gemini.yaml` with:
```yaml
import: definition.yaml

assignments:
  - agent: my-gemini
    task:
      - toolemu-std
      - toolemu-adv
      - toolemu-stress
      - toolemu-safety

concurrency:
  task:
    toolemu-std: 2
    toolemu-adv: 2
    toolemu-stress: 1
    toolemu-safety: 2
  agent:
    my-gemini: 2

output: "outputs/toolemu_{TIMESTAMP}"
```
Run it:
```powershell
python -m src.assigner --config configs/assignments/toolemu-gemini.yaml
```

- ALFWorld sample (dev single):
```powershell
python -m src.assigner --config configs/assignments/test_avalon.yaml
```

- DB + OS baseline with Gemini (optional):
Create `configs/assignments/combined-gemini.yaml` (example below in “Running Everything”). Then:
```powershell
python -m src.assigner --config configs/assignments/combined-gemini.yaml
```

5) Confirm output artifacts
- Outputs are under `outputs\*\{agent}\{task}\`
- Look for `runs.jsonl` and `overall.json`

6) Tear down the stack when finished:
```powershell
docker compose -f extra/docker-compose.yml down
```

---

## Running “Everything” in One Go (Custom Assignment)
If you want one command that includes multiple tasks at once with Gemini, create `configs/assignments/combined-gemini.yaml`:

```yaml
import: definition.yaml

concurrency:
  task:
    dbbench-std: 4
    os-std: 4
    alfworld-std: 1
    webshop-std: 1
    kg-std: 1
    toolemu-adv: 2
  agent:
    my-gemini: 4

assignments:
  - agent: my-gemini
    task:
      - dbbench-std
      - os-std
      - alfworld-std
      - webshop-std
      - kg-std
      - toolemu-adv

output: "outputs/{TIMESTAMP}"
```

Then run:
```powershell
python -m src.assigner --config configs/assignments/combined-gemini.yaml
```

Notes:
- Make sure the corresponding workers are running (Path A local workers, or Path B compose stack).
- You can add more tasks from `configs/tasks/task_assembly.yaml` (e.g., `mind2web`, `card_game`, `ltp`) if their environments are available.

---

## Troubleshooting (Windows)
- Use PowerShell as admin when needed for Docker operations.
- Ensure WSL2 backend is enabled in Docker Desktop settings.
- If ports 5000–5020 are in use, stop conflicting services or adjust:
  - Local run: `python -m src.server.task_controller -p <port>`
  - Update `controller_address` in `configs/assignments/definition.yaml` to match.
- KnowledgeGraph requires the Freebase data; without it, `kg-std` workers will fail to start or answer.
- WebShop needs ~16GB RAM to start stably.
- If `os_interaction` tasks fail to launch nested containers:
  - Verify `local-os/*` images exist on the host (`docker images`)
  - Ensure `/var/run/docker.sock` is mapped and reachable by the worker container (Compose does this).

Gemini-specific:
- Error 404 “models/… not found for API version v1beta”:
  - We default to `v1`. Ensure your base is `https://generativelanguage.googleapis.com/v1` (set `GEMINI_API_BASE`).
  - Try switching model between `gemini-2.5-pro` and `gemini-2.5-flash` (set `GEMINI_MODEL`).
  - Your account/region might not have access to a specific model alias; use `ListModels` in Google docs to confirm availability.

---

## Where Results Go
- Base folder: `outputs\{timestamp}\`
  - Per agent/task: `outputs\{timestamp}\{agent}\{task}\`
    - `runs.jsonl` – per-sample results
    - `error.jsonl` – failures
    - `overall.json` – aggregated metrics

---

## Short Reference
- Start local DB + OS workers:
  - `python -m src.start_task -a`
- Run DB + OS assignment:
  - For Gemini, use a custom assignment (e.g., `combined-gemini.yaml`)
- Start full stack:
  - `docker compose -f extra/docker-compose.yml up -d`
- Run ToolEmu suite:
  - `python -m src.assigner --config configs/assignments/toolemu-gemini.yaml`


