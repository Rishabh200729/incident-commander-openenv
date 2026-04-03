"""
FastAPI application for the Incident Commander Environment.

Exposes the OpenEnv-compliant HTTP API:
  POST /reset  — start a new episode
  POST /step   — execute an action
  GET  /state  — get current state
  GET  /health — health check

Compatible with Hugging Face Spaces deployment.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .environment import IncidentCommanderEnvironment
from .models import IncidentAction, IncidentObservation, IncidentState
from .tasks import list_tasks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models for the HTTP layer
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    task_name: Optional[str] = None


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False


class StepRequest(BaseModel):
    action: Dict[str, Any]


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False


class StateResponse(BaseModel):
    state: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str = "healthy"


class GradeResponse(BaseModel):
    score: float
    breakdown: Dict[str, float]
    steps_taken: int
    is_resolved: bool
    escalated: bool
    rewards: list


class TaskListResponse(BaseModel):
    tasks: list


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_incident_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Incident Commander Environment",
        description=(
            "An OpenEnv-compliant RL environment simulating production "
            "microservices incident response. The agent plays Incident "
            "Commander and must triage, diagnose, and resolve outages."
        ),
        version="1.0.0",
    )

    # CORS for Hugging Face Spaces
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Single environment instance (stateful) with concurrency guard
    env = IncidentCommanderEnvironment()
    _lock = asyncio.Lock()
    _is_initialised = False

    # ---- Health check ----
    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(status="healthy")

    # ---- Task list ----
    @app.get("/tasks", response_model=TaskListResponse)
    async def tasks():
        return TaskListResponse(tasks=list_tasks())

    # ---- Reset ----
    @app.post("/reset", response_model=ResetResponse)
    async def reset(request: ResetRequest = None):
        nonlocal _is_initialised
        req = request or ResetRequest()
        async with _lock:
            try:
                obs = env.reset(
                    seed=req.seed,
                    episode_id=req.episode_id,
                    task_name=req.task_name,
                )
                _is_initialised = True
                obs_dict = obs.model_dump()
                return ResetResponse(
                    observation=obs_dict,
                    reward=obs.reward,
                    done=obs.done,
                )
            except KeyError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Reset failed: {e}\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=str(e))

    # ---- Step ----
    @app.post("/step", response_model=StepResponse)
    async def step(request: StepRequest):
        if not _is_initialised:
            raise HTTPException(
                status_code=400,
                detail="Environment not initialised. Call POST /reset first.",
            )
        try:
            action = IncidentAction(**request.action)
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid action: {e}",
            )

        async with _lock:
            try:
                obs = env.step(action)
                obs_dict = obs.model_dump()
                return StepResponse(
                    observation=obs_dict,
                    reward=obs.reward,
                    done=obs.done,
                )
            except Exception as e:
                logger.error(f"Step failed: {e}\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=str(e))

    # ---- State ----
    @app.get("/state", response_model=StateResponse)
    async def state():
        return StateResponse(state=env.state.model_dump())

    # ---- Grade ----
    @app.get("/grade", response_model=GradeResponse)
    async def grade():
        result = env.grade()
        return GradeResponse(**result)

    # ---- Timeline (post-mortem) ----
    @app.get("/timeline")
    async def timeline():
        """Return the incident timeline for post-mortem analysis."""
        state = env.state
        return {
            "episode_id": state.episode_id,
            "task_name": state.task_name,
            "timeline": state.incident_timeline,
            "total_steps": state.step_count,
            "is_resolved": state.is_resolved,
        }

    # ---- Environment Info ----
    @app.get("/info")
    async def info():
        """Return environment metadata and capabilities."""
        from .services import DEPENDENCY_GRAPH, ALL_SERVICES
        from .tasks import list_tasks as _list_tasks, get_task
        tasks_info = {}
        for t_name in _list_tasks():
            t = get_task(t_name)
            tasks_info[t_name] = {
                "difficulty": t.difficulty,
                "max_steps": t.max_steps,
                "description": t.description,
            }
        return {
            "name": "incident_commander_env",
            "version": "1.0.0",
            "services": ALL_SERVICES,
            "dependency_graph": DEPENDENCY_GRAPH,
            "action_types": [
                "inspect_logs", "inspect_metrics",
                "restart_service", "scale_service",
                "rollback", "clear_cache",
                "escalate", "do_nothing",
            ],
            "tasks": tasks_info,
        }
    # ---- Metadata (required by openenv validate) ----
    @app.get("/metadata")
    async def metadata():
        """Return environment name and description for openenv validation."""
        return {
            "name": "incident_commander_env",
            "description": (
                "AI SRE Incident Commander — diagnose and resolve production "
                "microservices outages across 3 difficulty levels"
            ),
        }

    # ---- Schema (required by openenv validate) ----
    @app.get("/schema")
    async def schema():
        """Return JSON schemas for action, observation, and state models."""
        return {
            "action": IncidentAction.model_json_schema(),
            "observation": IncidentObservation.model_json_schema(),
            "state": IncidentState.model_json_schema(),
        }

    return app


# Create the app instance (used by uvicorn / openenv.yaml)
app = create_incident_app()


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# Alias for [project.scripts] entry point
run_server = main


if __name__ == "__main__":
    main()
