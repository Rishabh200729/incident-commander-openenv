"""
Pydantic models for the Incident Commander Environment.

Defines typed Action, Observation, and State models following the OpenEnv specification.
All models use Pydantic v2 for automatic validation and serialization.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """All valid actions the agent can take."""

    INSPECT_LOGS = "inspect_logs"
    INSPECT_METRICS = "inspect_metrics"
    RESTART_SERVICE = "restart_service"
    SCALE_SERVICE = "scale_service"
    ROLLBACK = "rollback"
    CLEAR_CACHE = "clear_cache"
    ESCALATE = "escalate"
    DO_NOTHING = "do_nothing"
    WRITE_RUNBOOK = "write_runbook"


class ServiceStatusEnum(str, Enum):
    """Health status of a single service."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


class SeverityLevel(str, Enum):
    """Incident severity classification."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    RESOLVED = "resolved"


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class IncidentAction(BaseModel):
    """
    A single action the Incident Commander agent can take.

    Attributes:
        action_type: The kind of action to execute.
        service_name: Target service (required for service-specific actions).
        metadata: Optional extra key-value pairs.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    action_type: ActionType = Field(
        ..., description="The type of action to execute"
    )
    service_name: Optional[str] = Field(
        default=None,
        description="Target service name (required for inspect_logs, "
        "inspect_metrics, restart_service, scale_service, rollback)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


# ---------------------------------------------------------------------------
# Observation sub-models
# ---------------------------------------------------------------------------

class ServiceState(BaseModel):
    """Observable state of a single micro-service."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Service identifier")
    status: ServiceStatusEnum = Field(..., description="Current health status")
    error_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of failed requests [0-1]"
    )
    latency_ms: float = Field(
        ..., ge=0.0, description="p95 response latency in milliseconds"
    )
    cpu_percent: float = Field(
        ..., ge=0.0, le=100.0, description="CPU utilisation percentage"
    )
    memory_percent: float = Field(
        ..., ge=0.0, le=100.0, description="Memory utilisation percentage"
    )
    instances: int = Field(
        default=1, ge=0, description="Number of running instances"
    )
    version: str = Field(
        default="v1.0.0", description="Currently deployed version"
    )
    log_quality: str = Field(
        default="full",
        description="Log availability quality: full | partial | empty | misleading",
    )


class IncidentObservation(BaseModel):
    """
    Full observation returned after reset() or step().

    Follows the OpenEnv Observation contract: includes ``done``, ``reward``,
    and ``metadata`` alongside domain-specific fields.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # --- OpenEnv standard fields ---
    done: bool = Field(default=False, description="Whether the episode has ended")
    reward: Optional[float] = Field(
        default=None, description="Reward from the last action"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    # --- Domain-specific fields ---
    services: Dict[str, ServiceState] = Field(
        default_factory=dict,
        description="Per-service observable state",
    )
    alerts: List[str] = Field(
        default_factory=list,
        description="Active alert messages",
    )
    logs: List[str] = Field(
        default_factory=list,
        description="Log entries returned by the latest inspect_logs action",
    )
    metrics_detail: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detailed metrics for a single service (after inspect_metrics)",
    )
    incident_severity: SeverityLevel = Field(
        default=SeverityLevel.LOW,
        description="Current incident severity level",
    )
    system_health_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Aggregate system health score [0=down, 1=fully healthy]",
    )
    step_count: int = Field(
        default=0, ge=0, description="Steps taken so far in this episode"
    )
    max_steps: int = Field(
        default=30, ge=1, description="Maximum steps allowed for this task"
    )
    last_action_error: Optional[str] = Field(
        default=None,
        description="Error message from the last action, or null",
    )
    task_name: str = Field(
        default="", description="Name of the current task"
    )
    runbook_memory: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Relevant past runbook entries for similar incidents",
    )
    escalation_tier: int = Field(
        default=1, ge=1, le=4,
        description="Current escalation tier (1=stable, 4=full cascade)",
    )
    services_at_risk: List[str] = Field(
        default_factory=list,
        description="Services about to degrade in the next step",
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class IncidentState(BaseModel):
    """
    Full environment state (superset of Observation).

    Returned by the ``state()`` endpoint. Includes internal tracking data
    not visible in observations.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    episode_id: Optional[str] = Field(
        default=None, description="Unique episode identifier"
    )
    step_count: int = Field(default=0, ge=0, description="Steps taken so far")
    task_name: str = Field(default="", description="Active task name")
    is_resolved: bool = Field(
        default=False, description="Whether the incident has been resolved"
    )
    cumulative_reward: float = Field(
        default=0.0, description="Sum of all rewards in the episode"
    )
    actions_taken: List[str] = Field(
        default_factory=list,
        description="History of action strings taken",
    )
    services: Dict[str, ServiceState] = Field(
        default_factory=dict,
        description="Per-service state snapshot",
    )
    incident_timeline: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Chronological event log for post-incident review",
    )

