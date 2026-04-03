# Decisions Log

Every design decision made during development, who made it, and why.

## Decisions Made Without User Approval

### D-01: MIT License
- REVERTED — removed after user flagged it

### D-02: Author Name — "Incident Commander Team"
- File: pyproject.toml — Change to your real name

### D-03: Default LLM Model — gpt-4o-mini
- File: inference.py — May not work with HuggingFace router

### D-04: Default API Endpoint — https://api.openai.com/v1
- Hackathon sample uses https://router.huggingface.co/v1

### D-05: 6 Microservices (database, cache, auth, notification, payments, checkout)
- Realistic e-commerce backend with enough complexity for 3 difficulty levels

### D-06: Health Score Weights
- database=25%, auth=20%, payments=20%, checkout=20%, cache=10%, notification=5%

### D-07: Reward Values
- inspect root cause: +0.05 | other: +0.02 | repeat: -0.01
- health delta x2.0 | correct recovery: +0.15 | wrong target: -0.03
- repeated action: -0.05 | do_nothing: -0.03 | completion: +0.20

### D-08: Grader Weights
- Recovery 40%, Efficiency 25%, Diagnostics 15%, Ordering 20%

### D-09: Max Steps — Easy=15, Medium=20, Hard=30

### D-10: Python >= 3.10

### D-11: Environment Name — incident_commander_env

### D-12: Port 8000

### D-13: Version v1.0.0 = healthy, anything else = bad deploy

### D-14: Dependency propagation is instant (same step)

### D-15: Bad deploys resist restart, require rollback

### D-16: Pydantic v2

### D-17: No heavy dependencies for core env

### D-18: Fully deterministic, no randomness

## Decisions Dictated by Hackathon/OpenEnv Spec

- FastAPI server, openenv.yaml, inference.py in root
- [START]/[STEP]/[END] stdout format, HF_TOKEN, score= field, flush=True
- OpenAI client, Docker, HF Spaces, 3+ tasks, scores 0.0-1.0
- [project.scripts] server, uv.lock, /metadata, /schema endpoints
