"""
Deterministic grader for the Incident Commander Environment.

Computes a final score in [0.0, 1.0] for each episode based on:
- System recovery (did the system return to healthy?)
- Efficiency (how quickly was it resolved?)
- Diagnostics (did the agent investigate before acting?)
- Correct ordering (were recovery actions in dependency order?)
- Memory utilization (did the agent leverage runbook data?)

Also provides per-step shaped rewards with revenue-loss escalation tiers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from .models import ServiceState, ServiceStatusEnum, ActionType
from .services import (
    ALL_SERVICES,
    DEPENDENCY_GRAPH,
    compute_health_score,
)
from .tasks import TaskDefinition


# ---------------------------------------------------------------------------
# Revenue loss per step by escalation tier (T2-8)
# ---------------------------------------------------------------------------

REVENUE_LOSS_PER_STEP = {
    1: -0.005,   # $5k/min — incident just started
    2: -0.015,   # $15k/min — spreading
    3: -0.030,   # $30k/min — cascading
    4: -0.060,   # $60k/min — full outage
}


# ---------------------------------------------------------------------------
# Per-step shaped reward
# ---------------------------------------------------------------------------

def compute_step_reward(
    prev_health: float,
    curr_health: float,
    action_type: str,
    service_name: Optional[str],
    task: TaskDefinition,
    actions_history: List[str],
    services: Dict[str, ServiceState],
    is_done: bool,
    steps_taken: int,
    escalation_tier: int = 1,
    runbook_used: bool = False,
    elapsed_seconds: float = 0.0,
    http_mode: bool = False,
) -> float:
    """
    Compute a shaped reward for a single step.

    Returns a float reward. Positive = good, negative = bad.

    Args:
        escalation_tier: Current escalation tier (1-4) for revenue loss penalty.
        runbook_used: True if agent's action matches a runbook fix suggestion.
        elapsed_seconds: Wall-clock seconds since episode start (HTTP mode only).
        http_mode: Whether the environment is running in HTTP server mode.
    """
    reward = 0.0

    # 1) Health improvement component (most important signal)
    health_delta = curr_health - prev_health
    reward += health_delta * 2.0  # scale up health changes

    # 2) Action-type bonuses / penalties
    action_str = f"{action_type}:{service_name}" if service_name else action_type

    if action_type in ("inspect_logs", "inspect_metrics"):
        # Diagnostic actions: small bonus for investigating (FIRST TIME ONLY)
        if service_name and service_name in ALL_SERVICES:
            # Check if this service was already inspected
            prior_inspections = sum(
                1 for a in actions_history[:-1]
                if a.endswith(f":{service_name}") and a.startswith("inspect_")
            )
            if prior_inspections == 0:
                # First inspection — reward
                if service_name == task.root_cause_service:
                    reward += 0.05
                else:
                    reward += 0.02
            else:
                reward -= 0.01  # penalize repeated inspection
        else:
            reward -= 0.01  # invalid service

    elif action_type in ("restart_service", "scale_service", "rollback"):
        if service_name:
            # Check if this is a correct recovery action
            if action_str in task.correct_recovery_actions:
                reward += 0.15
            elif service_name in ALL_SERVICES:
                # Valid service but not the ideal action
                svc = services.get(service_name)
                if svc and svc.status != ServiceStatusEnum.HEALTHY:
                    reward += 0.02  # at least targeting an unhealthy service
                else:
                    reward -= 0.03  # unnecessary action on healthy service

            # Penalize repeated recovery on same service
            same_action_count = sum(1 for a in actions_history if a == action_str)
            if same_action_count > 1:
                reward -= 0.05 * (same_action_count - 1)

    elif action_type == "clear_cache":
        if "clear_cache" in task.correct_recovery_actions:
            reward += 0.10
        else:
            reward -= 0.02  # unnecessary cache clear

    elif action_type == "escalate":
        # Partial credit — episode ends but agent gets credit for diagnostics done
        reward += 0.0  # neutral, final score handles partial credit

    elif action_type == "write_runbook":
        # Handled separately in environment — small neutral step cost
        reward += 0.0

    elif action_type == "do_nothing":
        if curr_health < 0.95:
            reward -= 0.03  # penalize wasting time during incident
        # If system is healthy, do_nothing is fine
        if curr_health >= 0.95:
            reward += 0.01

    # 3) Revenue loss penalty based on escalation tier (T2-8)
    # Only apply if this step is NOT a correct recovery action (Audit Fix #6)
    is_correct_recovery = action_str in task.correct_recovery_actions
    if curr_health < 0.95 and not is_correct_recovery:
        revenue_penalty = REVENUE_LOSS_PER_STEP.get(escalation_tier, -0.005)
        reward += revenue_penalty

    # 4) Time pressure penalty (HTTP mode only) (T2-5)
    if http_mode and elapsed_seconds > 0:
        reward += -0.001 * elapsed_seconds

    # 5) Runbook utilization bonus (T2-7)
    if runbook_used:
        reward += 0.08

    # 6) Completion bonus
    if is_done and curr_health >= 0.95:
        efficiency_bonus = max(0.0, (task.max_steps - steps_taken) / task.max_steps) * 0.3
        reward += 0.2 + efficiency_bonus

    return round(reward, 4)


# ---------------------------------------------------------------------------
# Final episode grading
# ---------------------------------------------------------------------------

def grade_episode(
    task: TaskDefinition,
    final_services: Dict[str, ServiceState],
    actions_history: List[str],
    steps_taken: int,
    is_resolved: bool,
    escalated: bool,
    runbook_written: bool = False,
    runbook_correct: bool = False,
    runbook_available: bool = False,
    runbook_used: bool = False,
    elapsed_seconds: float = 0.0,
    http_mode: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """
    Grade a completed episode.

    Returns:
        (score, breakdown) where score is in [0.0, 1.0] and breakdown
        is a dict of component scores for transparency.

    Args:
        runbook_written: Whether agent wrote a runbook this episode.
        runbook_correct: Whether the runbook contained correct root cause.
        runbook_available: Whether runbook data was available at episode start.
        runbook_used: Whether agent's first fix matched a runbook suggestion.
        elapsed_seconds: Wall-clock time elapsed (HTTP mode only).
        http_mode: Whether grading in HTTP server mode.
    """
    breakdown: Dict[str, float] = {}

    # --- Component 1: System Recovery (0-0.35) ---
    # (Reduced from 0.40 to accommodate memory utilization component)
    final_health = compute_health_score(final_services)
    healthy_count = sum(
        1 for s in final_services.values() if s.status == ServiceStatusEnum.HEALTHY
    )
    total = len(final_services)

    if is_resolved and final_health >= 0.95:
        recovery_score = 0.35
    elif final_health >= 0.80:
        recovery_score = 0.25
    elif final_health >= 0.60:
        recovery_score = 0.17
    elif final_health >= 0.40:
        recovery_score = 0.09
    else:
        recovery_score = 0.04 * (healthy_count / total)

    if escalated:
        recovery_score *= 0.5  # escalation caps recovery credit

    breakdown["recovery"] = round(recovery_score, 4)

    # --- Component 2: Efficiency (0-0.20) ---
    # (Reduced from 0.25 to accommodate memory utilization component)
    # Blends step-count and wall-clock efficiency in HTTP mode
    if is_resolved:
        optimal_steps = len(task.correct_recovery_actions) + 2  # allow some diagnostics

        # Step-count efficiency
        if steps_taken <= optimal_steps:
            step_efficiency = 1.0
        elif steps_taken <= optimal_steps * 2:
            ratio = 1.0 - (steps_taken - optimal_steps) / (optimal_steps)
            step_efficiency = max(0.0, ratio)
        else:
            step_efficiency = 0.2

        # Wall-clock efficiency (HTTP mode only)
        if http_mode and elapsed_seconds > 0:
            time_limit = task.time_limit_seconds
            time_efficiency = max(0.0, 1.0 - (elapsed_seconds / time_limit))
            # Blend 50/50 step + time efficiency
            combined_efficiency = 0.5 * time_efficiency + 0.5 * step_efficiency
        else:
            combined_efficiency = step_efficiency

        efficiency_score = 0.20 * combined_efficiency
    else:
        efficiency_score = 0.0

    breakdown["efficiency"] = round(efficiency_score, 4)

    # --- Component 3: Diagnostics (0-0.15) ---
    diagnostic_actions = [
        a for a in actions_history
        if a.startswith("inspect_logs:") or a.startswith("inspect_metrics:")
    ]
    investigated_services: Set[str] = set()
    for a in diagnostic_actions:
        parts = a.split(":", 1)
        if len(parts) == 2:
            investigated_services.add(parts[1])

    # Did the agent investigate the root cause service?
    found_root_cause = task.root_cause_service in investigated_services

    if found_root_cause and len(investigated_services) >= 2:
        diagnostics_score = 0.15
    elif found_root_cause:
        diagnostics_score = 0.10
    elif len(investigated_services) >= 2:
        diagnostics_score = 0.07
    elif len(investigated_services) >= 1:
        diagnostics_score = 0.04
    else:
        diagnostics_score = 0.0

    breakdown["diagnostics"] = round(diagnostics_score, 4)

    # --- Component 4: Correct Action Ordering (0-0.20) ---
    recovery_actions = [
        a for a in actions_history
        if not a.startswith("inspect_") and a != "do_nothing"
        and a != "escalate" and a != "write_runbook"
    ]

    if not task.correct_recovery_actions:
        order_score = 0.20 if is_resolved else 0.0
    else:
        # Check how many correct actions were taken in order
        correct_idx = 0
        matched = 0
        for action in recovery_actions:
            if correct_idx < len(task.correct_recovery_actions):
                if action == task.correct_recovery_actions[correct_idx]:
                    matched += 1
                    correct_idx += 1

        order_ratio = matched / len(task.correct_recovery_actions)
        order_score = 0.20 * order_ratio

    breakdown["ordering"] = round(order_score, 4)

    # --- Component 5: Memory Utilization (0-0.10) --- (T2-7)
    # Did the agent leverage available runbook data and contribute new knowledge?
    memory_score = 0.0

    if runbook_written and runbook_correct:
        # Agent wrote a useful runbook with correct root cause
        memory_score += 0.05
    elif runbook_written:
        # Wrote a runbook but wrong root cause
        memory_score += 0.02

    if runbook_available and runbook_used:
        # Agent successfully leveraged existing runbook knowledge
        memory_score += 0.05
    elif runbook_available:
        # Runbook was available but agent didn't use it — slight penalty
        memory_score += 0.0

    memory_score = min(0.10, memory_score)
    breakdown["memory"] = round(memory_score, 4)

    # --- Total ---
    total_score = sum(breakdown.values())
    total_score = round(min(1.0, max(0.0, total_score)), 4)

    return total_score, breakdown
