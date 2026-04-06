# 🚨 Enterprise Incident Commander Environment

An **OpenEnv-compliant** reinforcement learning environment where an AI agent plays the role of an **Incident Commander** for a microservices production system during a live outage.

The agent must triage alerts, inspect logs and metrics, identify the root cause, choose recovery actions, and restore the system to a healthy state.

> **New here?** Read [architecture.md](architecture.md) for a complete walkthrough — it explains everything from scratch, including RL concepts.

---

## 🎯 Motivation

Real-world production incidents are high-stakes, time-critical events that require systematic reasoning. This environment simulates realistic outage scenarios with:

- **Cascading failures** across dependent services
- **Misleading symptoms** that require deep investigation
- **Dependency-aware recovery** where order matters
- **Shaped rewards** that teach agents to diagnose before acting

---

## ⚡ Quick Start

### Prerequisites

- Python ≥ 3.10
- pip or uv

### 1. Install

```bash
git clone <repo-url>
cd incident-commander-env
pip install -e ".[dev]"
```

### 2. Run Tests (no API key needed)

```bash
python -m pytest tests/ -q
# Expected: 128 passed
```

### 3. Run Evaluation (no API key needed)

```bash
python evaluate.py
# Expected: 🎉 ALL CHECKS PASSED
```

### 4. Start the Server

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### 5. Run Inference (needs API key)

```bash
export HF_TOKEN="your-huggingface-or-openai-key"
export MODEL_NAME="gpt-4o-mini"
python inference.py
```

### 6. Docker

```bash
docker build -t incident-commander-env .
docker run -p 8000:8000 incident-commander-env
```

---

## 🏗️ Environment Design

### Microservices Architecture

```
┌──────────────┐     ┌──────────────┐
│   database   │     │    cache     │
│ (foundational)│     │ (foundational)│
└──────┬───┬───┘     └──────┬───────┘
       │   │                │
       │   │   ┌────────────┘
       │   │   │
  ┌────▼───▼───▼──┐     ┌──────────────┐
  │     auth      │     │ notification │
  │(db + cache)   │     │ (standalone) │
  └────┬──────────┘     └──────┬───────┘
       │                       │
  ┌────▼───────────────────────▼──┐
  │          payments             │
  │      (db + notification)     │
  └────────────┬──────────────────┘
               │
  ┌────────────▼──────────────────┐
  │          checkout             │
  │   (auth + payments + db)     │
  └───────────────────────────────┘
```

### Dependency Graph

| Service       | Dependencies              | Role             |
|---------------|---------------------------|------------------|
| `database`    | *(none)*                  | Foundational     |
| `cache`       | *(none)*                  | Foundational     |
| `auth`        | database, cache           | Authentication   |
| `notification`| *(none)*                  | Messaging        |
| `payments`    | database, notification    | Payment processing|
| `checkout`    | auth, payments, database  | User-facing      |

---

## 🎮 Action Space

8 action types available to the agent:

| Action             | Parameter       | Description                              |
|--------------------|-----------------|------------------------------------------|
| `inspect_logs`     | `service_name`  | View recent log entries for a service    |
| `inspect_metrics`  | `service_name`  | Get detailed metrics (CPU, latency, etc) |
| `restart_service`  | `service_name`  | Restart a service                        |
| `scale_service`    | `service_name`  | Add an instance to a service             |
| `rollback`         | `service_name`  | Roll back to the previous stable version |
| `clear_cache`      | *(none)*        | Flush the cache service                  |
| `escalate`         | *(none)*        | Escalate (ends episode, partial credit)  |
| `do_nothing`       | *(none)*        | Skip a step (penalized during incidents) |

### Action Format (JSON)

```json
{"action_type": "inspect_logs", "service_name": "auth"}
{"action_type": "clear_cache"}
```

---

## 👁️ Observation Space

Each observation contains:

| Field                | Type              | Description                           |
|----------------------|-------------------|---------------------------------------|
| `services`           | `Dict[str, ServiceState]` | Per-service health snapshot    |
| `alerts`             | `List[str]`       | Active alert messages                 |
| `logs`               | `List[str]`       | Logs from last `inspect_logs` action  |
| `metrics_detail`     | `Dict` or `null`  | Metrics from last `inspect_metrics`   |
| `incident_severity`  | `str`             | critical / high / medium / low / resolved |
| `system_health_score`| `float [0-1]`     | Aggregate weighted health score       |
| `step_count`         | `int`             | Current step number                   |
| `max_steps`          | `int`             | Maximum steps for this task           |
| `last_action_error`  | `str` or `null`   | Error from the last action            |
| `done`               | `bool`            | Whether the episode has ended         |
| `reward`             | `float`           | Shaped reward from the last step      |

---

## 📋 Tasks

### Task 1: Single Service Failure *(Easy)*

| Property      | Value                                              |
|---------------|----------------------------------------------------|
| Name          | `single_service_failure`                           |
| Max Steps     | 15                                                 |
| Root Cause    | Cache service crashed (OOM)                        |
| Fix           | `restart_service("cache")`                         |
| Optimal Steps | 2–3 (inspect → restart)                            |

### Task 2: Cascading Failure *(Medium)*

| Property      | Value                                              |
|---------------|----------------------------------------------------|
| Name          | `cascading_failure`                                |
| Max Steps     | 20                                                 |
| Root Cause    | Database overloaded causing cascading failures     |
| Fix           | Scale DB → restart DB → restart auth → restart payments → restart checkout |
| Optimal Steps | 6–8 (inspect → scale → restart chain)              |

### Task 3: Hidden Root Cause *(Hard)*

| Property      | Value                                              |
|---------------|----------------------------------------------------|
| Name          | `hidden_root_cause`                                |
| Max Steps     | 30                                                 |
| Root Cause    | Auth has bad deploy (v2.2.0-rc1) with JWT regression. Cache masks the issue. |
| Fix           | Rollback auth → clear cache → restart payments → restart checkout |
| Optimal Steps | 6–8 (inspect several services → rollback → clear cache → restart) |

---

## 💎 Reward Design

### Per-Step Shaped Rewards

| Signal                  | Reward      |
|-------------------------|-------------|
| Health improvement       | `delta * 2.0` |
| First inspect root cause | `+0.05`    |
| First inspect other      | `+0.02`    |
| Repeated inspect         | `-0.01`    |
| Correct recovery action  | `+0.15`    |
| Recovery on wrong target | `-0.03`    |
| Repeated same action     | `-0.05`    |
| `do_nothing` during outage| `-0.03`   |
| Episode completion bonus | `+0.2` + efficiency |

### Final Episode Score (Grader)

| Component   | Weight | Description                          |
|-------------|--------|--------------------------------------|
| Recovery    | 40%    | Did the system return to healthy?    |
| Efficiency  | 25%    | How quickly was it resolved?         |
| Diagnostics | 15%    | Did the agent investigate first?     |
| Ordering    | 20%    | Were actions in dependency order?    |

---

## 🧪 HTTP API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check → `{"status": "healthy"}` |
| GET | `/tasks` | List available tasks |
| GET | `/metadata` | Environment name + description |
| GET | `/schema` | JSON schemas for action/observation/state |
| POST | `/reset` | Start new episode |
| POST | `/step` | Execute an action |
| GET | `/state` | Get full environment state |
| GET | `/grade` | Get episode grade/score |
| GET | `/timeline` | Incident timeline for post-mortem |
| GET | `/info` | Environment metadata & capabilities |

### Example Usage

```bash
# Health check
curl http://localhost:8000/health

# List tasks
curl http://localhost:8000/tasks

# Reset environment
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "single_service_failure"}'

# Execute an action
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "inspect_logs", "service_name": "cache"}}'

# Get environment state
curl http://localhost:8000/state

# Get episode grade
curl http://localhost:8000/grade
```

### Python (Direct, no server)

```python
from server.environment import IncidentCommanderEnvironment
from server.models import IncidentAction, ActionType

env = IncidentCommanderEnvironment()
obs = env.reset(task_name="single_service_failure")

# Inspect logs
action = IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="cache")
obs = env.step(action)
print(obs.logs)

# Restart the failed service
action = IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="cache")
obs = env.step(action)
print(f"Health: {obs.system_health_score}, Done: {obs.done}")

# Get final grade
result = env.grade()
print(f"Score: {result['score']}")
```

---

## 🤖 Baseline Inference

### Configuration

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | *(required)* | Hugging Face / OpenAI API key |
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM API endpoint |
| `MODEL_NAME` | `gpt-4o-mini` | Model to use |

```bash
export HF_TOKEN="your-key"
python inference.py
```

### Expected Baseline Scores

| Task                    | Score Range  |
|-------------------------|-------------|
| single_service_failure  | 0.60 – 0.95 |
| cascading_failure       | 0.40 – 1.00 |
| hidden_root_cause       | 0.25 – 0.85 |

### Stdout Format

```
[START] task=<task_name> env=incident_commander_env model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
```

---

## ✅ Validation

```bash
# Validate project structure
openenv validate

# Run tests
python -m pytest tests/ -q

# Run evaluation
python evaluate.py
```

---

## 📁 Project Structure

```
├── openenv.yaml          # OpenEnv manifest
├── pyproject.toml         # Dependencies & config
├── uv.lock                # Locked dependencies
├── README.md              # This file
├── guide.md               # Detailed beginner-friendly guide
├── inference.py           # Baseline LLM inference script
├── evaluate.py            # Self-contained evaluation suite
├── Dockerfile             # Container for HF Spaces
├── __init__.py            # Package exports
├── server/
│   ├── __init__.py
│   ├── models.py          # Pydantic models (Action, Observation, State)
│   ├── services.py        # Microservice graph & simulation
│   ├── tasks.py           # 3 task definitions
│   ├── grader.py          # Scoring & reward calculation
│   ├── environment.py     # Core environment logic
│   └── app.py             # FastAPI HTTP server
└── tests/
    ├── conftest.py        # Shared fixtures
    ├── test_environment.py    # Core tests (48)
    ├── test_edge_cases.py     # Edge case tests (62)
    └── test_weakness_fixes.py # Regression tests (18)
```

---

## 🛠️ Contributing

### Development Setup

```bash
# Clone and install with dev dependencies
git clone <repo-url>
cd incident-commander-env
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run evaluation
python evaluate.py --verbose

# Start dev server
uvicorn server.app:app --reload --port 8000
```

### Adding a New Task

1. Open `server/tasks.py`
2. Create a new `_build_*_task()` function with initial service states
3. Register it in `TASK_REGISTRY`
4. Add expert + naive strategies in `evaluate.py`
5. Add tests in `tests/test_environment.py`
6. Update this README and `guide.md`

### Key Files to Understand

| File | What to read | When |
|---|---|---|
| `guide.md` | Full theory + architecture | Start here if new |
| `server/models.py` | Data types | Before touching any file |
| `server/services.py` | How simulation works | If changing game mechanics |
| `server/tasks.py` | Task definitions | If adding/modifying tasks |
| `server/grader.py` | Reward/scoring logic | If changing how agents are scored |
| `server/environment.py` | Core loop | If changing reset/step logic |

### Running Specific Tests

```bash
# Run only edge case tests
python -m pytest tests/test_edge_cases.py -v

# Run a specific test class
python -m pytest tests/test_environment.py::TestEasyTask -v

# Run with detailed output
python -m pytest tests/ -v --tb=long
```


