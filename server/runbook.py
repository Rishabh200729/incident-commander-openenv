"""
Runbook Memory — Retrieval-Augmented RL for Incident Commander.

Stores incident resolution patterns across episodes so the agent can build
institutional knowledge. The runbook is part of the observation space,
meaning the training loop can learn when to trust a runbook and how to
write better ones.

Pitch angle: "Our agent builds institutional knowledge across episodes —
just like a real SRE team."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RunbookEntry:
    """A single runbook entry recording how a past incident was resolved."""

    incident_type: str       # fingerprint: e.g. "database_oom" or "auth_bad_deploy"
    root_cause_service: str  # which service was the root cause
    fix_sequence: List[str]  # ordered list of actions that resolved the incident
    steps_taken: int         # how many steps the episode took
    score: float             # final episode score
    summary: str = ""        # agent-written summary of the incident
    episodes_ago: int = 0    # how many episodes ago this was written


class RunbookMemory:
    """
    Persistent cross-episode memory of incident resolution patterns.

    Stores up to `max_entries` runbook entries, most recent first.
    Provides lookup by incident fingerprint for retrieval-augmented
    observation injection.
    """

    def __init__(self, max_entries: int = 20) -> None:
        self.max_entries = max_entries
        self.entries: List[RunbookEntry] = []
        self._episode_counter: int = 0

    def lookup(self, incident_fingerprint: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Return top-k most relevant past runbooks by fingerprint similarity.

        Uses exact and partial matching on the incident fingerprint.
        Returns serialized dicts suitable for inclusion in observation.
        """
        # Score each entry by relevance to the query fingerprint
        scored: List[tuple[float, RunbookEntry]] = []

        for entry in self.entries:
            relevance = 0.0

            # Exact match on incident type
            if entry.incident_type == incident_fingerprint:
                relevance = 1.0
            else:
                # Partial match: check if root cause service or failure mode overlap
                query_parts = set(incident_fingerprint.lower().split("_"))
                entry_parts = set(entry.incident_type.lower().split("_"))
                overlap = query_parts & entry_parts
                if overlap:
                    relevance = len(overlap) / max(len(query_parts), len(entry_parts))

            if relevance > 0.0:
                scored.append((relevance, entry))

        # Sort by relevance (descending), then by recency (fewer episodes ago)
        scored.sort(key=lambda x: (-x[0], x[1].episodes_ago))

        results = []
        for _, entry in scored[:top_k]:
            results.append({
                "incident_type": entry.incident_type,
                "root_cause_service": entry.root_cause_service,
                "fix_sequence": entry.fix_sequence,
                "steps_taken": entry.steps_taken,
                "score": round(entry.score, 3),
                "episodes_ago": entry.episodes_ago,
                "summary": entry.summary,
            })

        return results

    def write(self, entry: RunbookEntry) -> None:
        """
        Add a new runbook entry. Evicts oldest entry if at capacity.
        """
        self.entries.insert(0, entry)
        if len(self.entries) > self.max_entries:
            self.entries.pop()

    def advance_episode(self) -> None:
        """
        Increment episode counter for all entries (called at reset).
        """
        self._episode_counter += 1
        for entry in self.entries:
            entry.episodes_ago += 1

    def build_fingerprint(self, root_cause_service: str, task_name: str) -> str:
        """
        Build an incident fingerprint from task metadata.

        Examples: "database_oom", "auth_bad_deploy", "random_cache_cpu_spike"
        """
        if task_name == "single_service_failure":
            return f"{root_cause_service}_oom"
        elif task_name == "cascading_failure":
            return f"{root_cause_service}_overload"
        elif task_name == "hidden_root_cause":
            return f"{root_cause_service}_bad_deploy"
        elif task_name == "chaos_cascade":
            return f"{root_cause_service}_chaos"
        elif task_name == "multi_root_cause":
            return f"{root_cause_service}_multi"
        elif task_name == "random_incident":
            # For random incidents, the description contains the failure mode
            return f"random_{root_cause_service}"
        else:
            return f"{root_cause_service}_{task_name}"

    @property
    def size(self) -> int:
        """Number of stored runbook entries."""
        return len(self.entries)
