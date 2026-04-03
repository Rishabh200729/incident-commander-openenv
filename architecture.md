# 🏗️ Architecture — Enterprise Incident Commander Environment

> Complete technical and conceptual breakdown of the project.  
> No ML/RL background required — everything is explained from first principles.

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [Reinforcement Learning in 5 Minutes](#2-reinforcement-learning-in-5-minutes)
3. [What is OpenEnv?](#3-what-is-openenv)
4. [System Architecture](#4-system-architecture)
5. [Microservices Simulation](#5-microservices-simulation)
6. [The 3 Tasks](#6-the-3-tasks)
7. [Actions — What the Agent Can Do](#7-actions)
8. [Observations — What the Agent Sees](#8-observations)
9. [Reward Design](#9-reward-design)
10. [The Grader](#10-the-grader)
11. [Code Walkthrough](#11-code-walkthrough)
12. [Inference Script](#12-inference-script)
13. [Data Flow — Start to Finish](#13-data-flow)
14. [Deployment](#14-deployment)
15. [Glossary](#15-glossary)

---

## 1. What This Project Is

Imagine you're an on-call engineer at a big tech company. At 3 AM, your phone buzzes:

> *"CRITICAL: Checkout service is down. 10,000 users affected."*

You would:
1. **Look at dashboards** — which services are broken?
2. **Read logs** — what error messages are appearing?
3. **Think** — is checkout broken because of itself, or because something it *depends on* (like the database) is broken?
4. **Act** — restart the right service, scale it up, or roll back a bad deployment
5. **Verify** — is everything healthy again?

**This project simulates that exact scenario.** Instead of a human, an **AI agent** responds to the outage. It gets the same information and takes the same actions, and we score how well it does.

---

## 2. Reinforcement Learning in 5 Minutes

Reinforcement Learning (RL) is how AI learns by **trial and feedback**.

### The Dog Analogy

```
Dog sees ball → Fetches it → Gets a treat       (+reward)
Dog sees ball → Ignores it → No treat            (0 reward)
Dog sees shoe → Chews it   → Gets scolded        (-reward)
```

The dog learns: *fetch = good*, *chew shoe = bad*.

### Same Idea, Different Names

| Real World | RL Term | In Our Project |
|---|---|---|
| The dog | **Agent** | An LLM (like GPT-4) |
| The world | **Environment** | Our microservices simulation |
| What the dog sees | **Observation** | Service statuses, alerts, logs |
| What the dog does | **Action** | `restart_service`, `inspect_logs`, etc. |
| The treat / scolding | **Reward** | A number saying "good move" or "bad move" |
| One fetch session | **Episode** | One complete incident response attempt |

### The RL Loop

```
┌─────────────────────────────────────────────────┐
│  1. Agent sees observation                      │
│  2. Agent picks an action                       │
│  3. Environment processes action                │
│  4. Environment returns: observation + reward    │
│  5. Repeat until done = true                    │
└─────────────────────────────────────────────────┘
```

That's all of RL. Our project is the *environment* half of this loop.

---

## 3. What is OpenEnv?

**OpenEnv** is a standard by Meta for building RL environments. It's a *contract*:

> "Build your environment with these exact functions, and any AI agent can use it."

### The 3 Required Functions

| Function | Purpose | Analogy |
|---|---|---|
| `reset()` | Start a new episode. Returns initial observation. | "New game" |
| `step(action)` | Execute an action. Returns new observation + reward + done. | "Play a move" |
| `state()` | Get internal state. | "Look behind the curtain" |

### OpenEnv Also Requires

- **FastAPI HTTP server** exposing these as web endpoints
- **`openenv.yaml`** manifest describing the environment
- **Dockerfile** for containerized deployment
- Deployment to **Hugging Face Spaces**

---

## 4. System Architecture

### High-Level View

```
┌─────────────────────────────────────────────────────────┐
│                    INFERENCE (Agent)                     │
│                                                         │
│   inference.py  ──→  LLM (GPT-4)  ──→  Action JSON     │
└────────────────────────┬────────────────────────────────┘
                         │ calls step(action)
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    HTTP LAYER                            │
│                                                         │
│   app.py  (FastAPI)                                     │
│   POST /reset  │  POST /step  │  GET /state  │  GET /grade
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    CORE ENVIRONMENT                      │
│                                                         │
│   environment.py (IncidentCommanderEnvironment)         │
│   • reset()  • step()  • grade()  • state              │
└───────┬──────────────┬──────────────┬───────────────────┘
        │              │              │
        ▼              ▼              ▼
  ┌───────────┐  ┌───────────┐  ┌───────────┐
  │services.py│  │ tasks.py  │  │ grader.py │
  │Simulation │  │ Scenarios │  │  Scoring  │
  │  Engine   │  │           │  │           │
  └─────┬─────┘  └───────────┘  └───────────┘
        │
        ▼
  ┌───────────┐
  │ models.py │
  │   Types   │
  └───────────┘
```

### File Responsibilities

| File | Role | Lines |
|---|---|---|
| `models.py` | Pydantic data models — `Action`, `Observation`, `State`, `ServiceState` | ~205 |
| `services.py` | Simulation physics — logs, metrics, health, dependency propagation | ~443 |
| `tasks.py` | 3 task definitions with initial broken states | ~263 |
| `grader.py` | Per-step reward calculation + final 4-component scoring | ~230 |
| `environment.py` | Core `reset()`/`step()`/`grade()` logic, ties everything together | ~340 |
| `app.py` | FastAPI HTTP server wrapping the environment | ~265 |
| `inference.py` | LLM agent that plays the environment | ~425 |
| `evaluate.py` | Self-contained test suite (no API key needed) | ~367 |

---

## 5. Microservices Simulation

### The 6 Services

```
┌──────────────┐     ┌──────────────┐
│   DATABASE   │     │    CACHE     │
│ (stores data)│     │(fast lookups)│
└──────┬───┬───┘     └──────┬───────┘
       │   │                │
       │   └────────┬───────┘
       │            │
  ┌────▼────────────▼──┐     ┌──────────────┐
  │       AUTH         │     │ NOTIFICATION │
  │ (login checks)     │     │  (send SMS)  │
  └────┬───────────────┘     └──────┬───────┘
       │                            │
  ┌────▼────────────────────────────▼──┐
  │            PAYMENTS                │
  │      (process charges)             │
  └────────────┬───────────────────────┘
               │
  ┌────────────▼───────────────────────┐
  │            CHECKOUT                │
  │       (user-facing page)           │
  └────────────────────────────────────┘
```

### Dependency Graph

| Service | Depends On | Health Weight |
|---|---|---|
| `database` | *(none)* | 25% |
| `cache` | *(none)* | 10% |
| `auth` | database, cache | 20% |
| `notification` | *(none)* | 5% |
| `payments` | database, notification | 20% |
| `checkout` | auth, payments, database | 20% |

### Service Properties

Each service has real production metrics:

| Property | Range | Meaning |
|---|---|---|
| `status` | healthy / degraded / down | Overall health |
| `error_rate` | 0.0–1.0 | % of requests failing (0.01 = normal) |
| `latency_ms` | 0–∞ | Response time (20ms = fast, 5000ms = slow) |
| `cpu_percent` | 0–100 | CPU usage (>90% = overloaded) |
| `memory_percent` | 0–100 | RAM usage |
| `instances` | 0–N | Running copies (0 = dead) |
| `version` | string | Deployed code version |

### Dependency Propagation — The Key Mechanic

When you fix a service, its dependents **auto-heal** (unless they have their own problem):

```
Database: DOWN → restart → HEALTHY
    ↓ propagation
Auth: DEGRADED → dependency fixed → HEALTHY (auto)
    ↓ propagation  
Checkout: DOWN → dependency fixed → HEALTHY (auto)
```

**But**: if a service has an *intrinsic fault* (like bad code), restart won't work. You need `rollback`.

---

## 6. The 3 Tasks

### Task 1: Single Service Failure *(Easy — 15 steps)*

| | |
|---|---|
| **What's broken** | Cache crashed (OOM — Out of Memory) |
| **Symptoms** | Cache DOWN, Auth DEGRADED, Checkout DEGRADED |
| **The fix** | Restart cache → everything auto-heals |
| **Optimal** | 2 steps: inspect → restart |
| **Teaches** | Basic diagnosis and recovery |

### Task 2: Cascading Failure *(Medium — 20 steps)*

| | |
|---|---|
| **What's broken** | Database overloaded (92% CPU, high latency) |
| **Symptoms** | Database DEGRADED, Auth DEGRADED, Payments DEGRADED, Checkout DOWN |
| **The fix** | Scale DB → restart DB → restart auth → payments → checkout |
| **Optimal** | 6–8 steps |
| **Teaches** | Cascading failures, dependency ordering |

### Task 3: Hidden Root Cause *(Hard — 30 steps)*

| | |
|---|---|
| **What's broken** | Auth deployed bad code (v2.2.0-rc1), JWT tokens fail. Cache serves stale tokens, masking it. |
| **Symptoms** | Auth DEGRADED (version v2.2.0-rc1), Payments DEGRADED, Checkout DEGRADED |
| **The fix** | Rollback auth → clear cache → restart payments → checkout |
| **Why hard** | Restart does nothing (bad code re-deploys). Version string is the only clue. |
| **Teaches** | Deep investigation, bad deploys, caching red herrings |

---

## 7. Actions

| Action | Needs Service? | Effect |
|---|---|---|
| `inspect_logs` | ✅ | Read service logs. Shows errors, stack traces. |
| `inspect_metrics` | ✅ | Detailed CPU, memory, latency, dependency health. |
| `restart_service` | ✅ | Restart. Fixes most problems. Doesn't fix bad deploys. |
| `scale_service` | ✅ | Add an instance. Reduces CPU/memory pressure. |
| `rollback` | ✅ | Roll back to v1.0.0. Fixes bad deployments. |
| `clear_cache` | ❌ | Flush all cached data. Clears stale entries. |
| `escalate` | ❌ | Give up, call a human. Partial credit. Episode ends. |
| `do_nothing` | ❌ | Skip. Penalized during active incidents. |

```json
{"action_type": "restart_service", "service_name": "database"}
{"action_type": "clear_cache"}
```

---

## 8. Observations

After each action, the agent receives:

```json
{
  "services": { "database": {"status": "healthy", ...}, ... },
  "alerts": ["CRITICAL: cache is DOWN"],
  "logs": ["[ERROR] OOM killed process 1234"],
  "metrics_detail": { "cpu_percent": 92.0, ... },
  "incident_severity": "critical",
  "system_health_score": 0.55,
  "step_count": 3,
  "max_steps": 15,
  "last_action_error": null,
  "done": false,
  "reward": 0.05
}
```

---

## 9. Reward Design

### Why Shaped Rewards Matter

**Bad (sparse):** You get +1 if you win, 0 if you lose. After 15 random actions and scoring 0, the agent has no idea which actions were bad.

**Good (shaped — what we do):** Every single action gets immediate feedback:

| Signal | Reward | Reasoning |
|---|---|---|
| First inspect of root cause | +0.05 | Investigating the actual problem |
| First inspect of other service | +0.02 | Any investigation helps |
| Repeated inspect of same service | -0.01 | Stop wasting time |
| Health improvement | delta × 2.0 | **Biggest signal** — actually fixing things |
| Correct recovery action | +0.15 | Right fix on right target |
| Recovery on wrong target | -0.03 | Fixing the wrong thing |
| Repeated same action | -0.05 | Don't spam |
| `do_nothing` during outage | -0.03 | There's a crisis, act! |
| Episode completion bonus | +0.20 | Resolved the incident |

The agent knows **immediately** which actions help and which don't.

---

## 10. The Grader

Final score (0.0–1.0) from 4 components:

| Component | Weight | Measures |
|---|---|---|
| **Recovery** | 40% | Did the system get fixed? |
| **Efficiency** | 25% | How fast? (fewer steps = better) |
| **Diagnostics** | 15% | Did agent investigate before acting? |
| **Ordering** | 20% | Actions in dependency order? |

### Score Comparison

| Strategy | Easy | Medium | Hard |
|---|---|---|---|
| Expert (optimal) | 0.95 | 1.00 | 0.85 |
| Naive (restart all) | 0.85 | 0.65 | 0.20 |
| Do Nothing | 0.10 | 0.02 | 0.10 |
| Fallback (no LLM) | 0.92 | 0.73 | 0.85 |

Expert > Naive > DoNothing on every task — proves the grading works.

---

## 11. Code Walkthrough

### `models.py` — Data Types

Uses **Pydantic** (Python library for typed data validation):

- **`ActionType`** — enum of 8 action types
- **`ServiceStatusEnum`** — healthy / degraded / down
- **`IncidentAction`** — `{action_type, service_name?, metadata?}`
- **`ServiceState`** — all service metrics
- **`IncidentObservation`** — everything the agent sees
- **`IncidentState`** — internal tracking + timeline

### `services.py` — Simulation Engine

The "physics" of our world:

- **`DEPENDENCY_GRAPH`** — which services depend on which
- **`generate_logs()`** — creates realistic log messages per service status
- **`generate_metrics()`** — creates detailed metric snapshots
- **`compute_health_score()`** — weighted average across services
- **`propagate_dependencies()`** — **the key function** — auto-heals dependents when deps are fixed
- **`apply_restart/scale/rollback/clear_cache()`** — state transition functions

### `tasks.py` — Scenario Definitions

3 frozen (immutable) `TaskDefinition` objects:
- Initial service states (which services are broken and how)
- Root cause metadata (for the grader)
- Correct recovery action sequence (for ordering scoring)

### `grader.py` — Scoring Engine

- **`compute_step_reward()`** — called after every action, returns shaped reward
- **`grade_episode()`** — called at the end, returns 4-component score

Anti-exploit: repeated inspections penalized after the first.

### `environment.py` — The Brain

`IncidentCommanderEnvironment` class:

```python
env = IncidentCommanderEnvironment()
obs = env.reset(task_name="single_service_failure")    # start
obs = env.step(IncidentAction(...))                     # act
state = env.state                                       # check
grade = env.grade()                                     # score
```

### `app.py` — HTTP Layer

FastAPI server with these endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check |
| `/tasks` | GET | List tasks |
| `/metadata` | GET | Name + description (OpenEnv) |
| `/schema` | GET | JSON schemas (OpenEnv) |
| `/reset` | POST | Start episode |
| `/step` | POST | Execute action |
| `/state` | GET | Get state |
| `/grade` | GET | Get score |
| `/timeline` | GET | Incident timeline |
| `/info` | GET | Env metadata |

Includes: asyncio lock (concurrency), 400 guards (step before reset), CORS.

---

## 12. Inference Script

`inference.py` — the AI agent that *plays* the environment.

### Flow

```
for each task:
    1. Print [START]
    2. reset(task)
    3. Loop:
       a. Convert observation → text prompt (with action history)
       b. Ask LLM: "What should we do next?"
       c. Parse LLM response → JSON action
       d. If LLM fails → use smart fallback agent
       e. step(action)
       f. Print [STEP]
    4. grade()
    5. Print [END]
```

### Smart Fallback (no LLM needed)

When LLM returns garbage, a deterministic fallback takes over:

1. **Inspect** un-inspected broken services
2. **Rollback** services with non-standard versions
3. **Restart** the worst un-restarted service
4. **Scale** remaining broken services
5. **do_nothing** as last resort

This alone solves all 3 tasks (0.73–0.92)!

### Required Stdout Format

```
[START] task=single_service_failure env=incident_commander_env model=gpt-4o-mini
[STEP] step=1 action=inspect_logs:cache reward=0.05 done=false error=null
[STEP] step=2 action=restart_service:cache reward=1.47 done=true error=null
[END] success=true steps=2 score=0.950 rewards=0.05,1.47
```

---

## 13. Data Flow — Start to Finish

### Episode Lifecycle

```
User/Agent                    Environment
    │                              │
    ├── POST /reset ──────────────→│ Load task, set service states
    │                              │ Return initial observation
    │←─── observation ─────────────┤
    │                              │
    ├── POST /step ───────────────→│ Validate action
    │   {action_type, service}     │ Execute (restart/inspect/etc)
    │                              │ Propagate dependencies
    │                              │ Compute reward
    │                              │ Check if done
    │←─── observation + reward ────┤
    │                              │
    │    ... repeat steps ...       │
    │                              │
    ├── GET /grade ───────────────→│ Compute 4-component score
    │←─── {score, breakdown} ──────┤
```

### Dependency Propagation Detail

```
Before:  database=DOWN  auth=DEGRADED  checkout=DOWN

Action: restart_service("database")

Step 1: database → HEALTHY  (direct effect)
Step 2: propagate → auth depends on [database ✅, cache ✅] → auth → HEALTHY
Step 3: propagate → checkout depends on [auth ✅, payments, database ✅] → ...
Step 4: propagate → if all deps healthy AND no intrinsic fault → service → HEALTHY

After:   database=HEALTHY  auth=HEALTHY  checkout=depends...
```

---

## 14. Deployment

### Local Development

```bash
pip install -e ".[dev]"           # install with dev deps
python -m pytest tests/ -q        # run tests
python evaluate.py                # run evaluation
uvicorn server.app:app --reload   # start dev server
```

### Docker

```bash
docker build -t incident-commander-env .
docker run -p 8000:8000 incident-commander-env
```

### Hugging Face Spaces

1. Create Space at huggingface.co → SDK: Docker
2. Push this repo
3. Dockerfile auto-builds
4. Live at `https://your-space.hf.space`

### Validation

```bash
openenv validate                  # project structure ✅
python -m pytest tests/ -q        # 128 tests ✅
python evaluate.py                # scoring ✅
```

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | For inference | — | API key (HuggingFace or OpenAI) |
| `API_BASE_URL` | No | `https://api.openai.com/v1` | LLM endpoint |
| `MODEL_NAME` | No | `gpt-4o-mini` | LLM model |

---

## 15. Glossary

| Term | Meaning |
|---|---|
| **Agent** | The AI interacting with the environment |
| **Environment** | The simulation the agent interacts with |
| **Episode** | One complete incident response attempt (reset → done) |
| **Observation** | What the agent sees after each action |
| **Action** | What the agent does |
| **Reward** | Number indicating how good/bad an action was |
| **State** | Internal tracking data |
| **Step** | One action + result |
| **Grader** | Computes final score (0.0–1.0) |
| **Dependency propagation** | Fixing service A auto-fixes services depending on A |
| **Intrinsic fault** | Problem surviving restart (e.g., bad deploy) |
| **Reward shaping** | Giving intermediate rewards, not just end-of-episode |
| **Cascading failure** | One service breaking causes chain of failures |
| **Pydantic** | Python typed data validation library |
| **FastAPI** | Python web framework |
| **OpenEnv** | Meta's standard for RL environments |
| **SRE** | Site Reliability Engineering |
| **OOM** | Out of Memory |
| **JWT** | JSON Web Token (authentication) |
