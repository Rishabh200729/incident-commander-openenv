"""
Task definitions for the Incident Commander Environment.

Each task defines:
- initial service states (the "broken" cluster)
- task metadata (name, max_steps, description)
- root cause information for the grader
- the sequence of correct recovery actions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .models import ServiceState, ServiceStatusEnum


@dataclass(frozen=True)
class TaskDefinition:
    """Immutable description of a single task/scenario."""

    name: str
    description: str
    difficulty: str  # "easy", "medium", "hard"
    max_steps: int
    root_cause_service: str
    root_cause_description: str
    correct_recovery_actions: List[str]   # ordered list of action strings
    initial_services: Dict[str, ServiceState]


# ---------------------------------------------------------------------------
# Task 1 — Easy: Single Service Failure
# ---------------------------------------------------------------------------

def _build_easy_task() -> TaskDefinition:
    """Cache is DOWN → auth is DEGRADED. Fix: restart cache."""
    services = {
        "database": ServiceState(
            name="database", status=ServiceStatusEnum.HEALTHY,
            error_rate=0.01, latency_ms=20.0,
            cpu_percent=12.0, memory_percent=28.0,
            instances=2, version="v1.0.0",
        ),
        "cache": ServiceState(
            name="cache", status=ServiceStatusEnum.DOWN,
            error_rate=1.0, latency_ms=0.0,
            cpu_percent=0.0, memory_percent=0.0,
            instances=0, version="v1.0.0",
        ),
        "auth": ServiceState(
            name="auth", status=ServiceStatusEnum.DEGRADED,
            error_rate=0.40, latency_ms=2500.0,
            cpu_percent=65.0, memory_percent=55.0,
            instances=2, version="v1.0.0",
        ),
        "notification": ServiceState(
            name="notification", status=ServiceStatusEnum.HEALTHY,
            error_rate=0.01, latency_ms=30.0,
            cpu_percent=10.0, memory_percent=20.0,
            instances=1, version="v1.0.0",
        ),
        "payments": ServiceState(
            name="payments", status=ServiceStatusEnum.HEALTHY,
            error_rate=0.02, latency_ms=40.0,
            cpu_percent=18.0, memory_percent=32.0,
            instances=2, version="v1.0.0",
        ),
        "checkout": ServiceState(
            name="checkout", status=ServiceStatusEnum.DEGRADED,
            error_rate=0.30, latency_ms=3000.0,
            cpu_percent=55.0, memory_percent=50.0,
            instances=2, version="v1.0.0",
        ),
    }
    return TaskDefinition(
        name="single_service_failure",
        description=(
            "A single service (cache) has crashed, causing degraded performance "
            "in dependent services. Identify the failed service and restart it."
        ),
        difficulty="easy",
        max_steps=15,
        root_cause_service="cache",
        root_cause_description="Cache service OOM crash — needs restart",
        correct_recovery_actions=[
            "restart_service:cache",
        ],
        initial_services=services,
    )


# ---------------------------------------------------------------------------
# Task 2 — Medium: Cascading Failure
# ---------------------------------------------------------------------------

def _build_medium_task() -> TaskDefinition:
    """Database overloaded → auth, payments, checkout cascade."""
    services = {
        "database": ServiceState(
            name="database", status=ServiceStatusEnum.DEGRADED,
            error_rate=0.25, latency_ms=4500.0,
            cpu_percent=92.0, memory_percent=85.0,
            instances=2, version="v1.0.0",
        ),
        "cache": ServiceState(
            name="cache", status=ServiceStatusEnum.HEALTHY,
            error_rate=0.01, latency_ms=15.0,
            cpu_percent=10.0, memory_percent=25.0,
            instances=2, version="v1.0.0",
        ),
        "auth": ServiceState(
            name="auth", status=ServiceStatusEnum.DEGRADED,
            error_rate=0.30, latency_ms=3200.0,
            cpu_percent=70.0, memory_percent=60.0,
            instances=2, version="v1.0.0",
        ),
        "notification": ServiceState(
            name="notification", status=ServiceStatusEnum.HEALTHY,
            error_rate=0.02, latency_ms=25.0,
            cpu_percent=8.0, memory_percent=18.0,
            instances=1, version="v1.0.0",
        ),
        "payments": ServiceState(
            name="payments", status=ServiceStatusEnum.DEGRADED,
            error_rate=0.35, latency_ms=5000.0,
            cpu_percent=75.0, memory_percent=65.0,
            instances=2, version="v1.0.0",
        ),
        "checkout": ServiceState(
            name="checkout", status=ServiceStatusEnum.DOWN,
            error_rate=0.85, latency_ms=8000.0,
            cpu_percent=95.0, memory_percent=90.0,
            instances=2, version="v1.0.0",
        ),
    }
    return TaskDefinition(
        name="cascading_failure",
        description=(
            "The database is under heavy load, causing cascading failures across "
            "auth, payments, and checkout. Scale the database first, then restart "
            "dependent services in the correct dependency order."
        ),
        difficulty="medium",
        max_steps=20,
        root_cause_service="database",
        root_cause_description="Database overloaded — needs scaling and restart before dependents can recover",
        correct_recovery_actions=[
            "scale_service:database",
            "restart_service:database",
            "restart_service:auth",
            "restart_service:payments",
            "restart_service:checkout",
        ],
        initial_services=services,
    )


# ---------------------------------------------------------------------------
# Task 3 — Hard: Hidden Root Cause
# ---------------------------------------------------------------------------

def _build_hard_task() -> TaskDefinition:
    """
    Symptoms in checkout/payments. Red herring: payments looks broken.
    True root cause: auth has bad deploy v2.2.0-rc1 causing 401 errors.
    Cache is serving stale tokens, masking the issue intermittently.
    """
    services = {
        "database": ServiceState(
            name="database", status=ServiceStatusEnum.HEALTHY,
            error_rate=0.01, latency_ms=22.0,
            cpu_percent=14.0, memory_percent=30.0,
            instances=2, version="v1.0.0",
        ),
        "cache": ServiceState(
            name="cache", status=ServiceStatusEnum.HEALTHY,
            error_rate=0.02, latency_ms=12.0,
            cpu_percent=15.0, memory_percent=35.0,
            instances=2, version="v1.0.0",
        ),
        "auth": ServiceState(
            name="auth", status=ServiceStatusEnum.DEGRADED,
            error_rate=0.32, latency_ms=600.0,
            cpu_percent=45.0, memory_percent=40.0,
            instances=2, version="v2.2.0-rc1",  # <-- BAD DEPLOY
        ),
        "notification": ServiceState(
            name="notification", status=ServiceStatusEnum.HEALTHY,
            error_rate=0.01, latency_ms=20.0,
            cpu_percent=8.0, memory_percent=15.0,
            instances=1, version="v1.0.0",
        ),
        "payments": ServiceState(
            name="payments", status=ServiceStatusEnum.DEGRADED,
            error_rate=0.38, latency_ms=2800.0,
            cpu_percent=60.0, memory_percent=55.0,
            instances=2, version="v1.0.0",
        ),
        "checkout": ServiceState(
            name="checkout", status=ServiceStatusEnum.DEGRADED,
            error_rate=0.45, latency_ms=4000.0,
            cpu_percent=70.0, memory_percent=60.0,
            instances=2, version="v1.0.0",
        ),
    }
    return TaskDefinition(
        name="hidden_root_cause",
        description=(
            "Checkout and payments are experiencing high error rates. "
            "The symptoms are misleading — investigate carefully to find "
            "the true root cause before taking recovery actions."
        ),
        difficulty="hard",
        max_steps=30,
        root_cause_service="auth",
        root_cause_description=(
            "Auth service deployed v2.2.0-rc1 with JWT signature algorithm mismatch. "
            "Cache is serving stale tokens masking the issue intermittently."
        ),
        correct_recovery_actions=[
            "rollback:auth",
            "clear_cache",
            "restart_service:payments",
            "restart_service:checkout",
        ],
        initial_services=services,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, TaskDefinition] = {}


def _register_tasks() -> None:
    global TASK_REGISTRY
    TASK_REGISTRY = {
        "single_service_failure": _build_easy_task(),
        "cascading_failure": _build_medium_task(),
        "hidden_root_cause": _build_hard_task(),
    }


_register_tasks()


def get_task(name: str) -> TaskDefinition:
    """Retrieve a task definition by name. Raises KeyError if not found."""
    if name not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task '{name}'. Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[name]


def list_tasks() -> List[str]:
    """Return list of available task names."""
    return list(TASK_REGISTRY.keys())
