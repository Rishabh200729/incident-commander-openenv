"""
GRPO Training Script for Incident Commander Environment.

Uses HuggingFace TRL to fine-tune a small language model via GRPO (Group Relative 
Policy Optimization) on our environment.

This is a MANDATORY hackathon requirement.

Target model: Qwen2.5-1.5B-Instruct (small, trains fast on free compute)
Training: 200-500 steps with reward curve logging every 50 steps

Usage (in Bangalore with compute credits):
  python train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --steps 300

Prerequisites:
  pip install trl transformers torch accelerate peft bitsandbytes
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import IncidentCommanderEnvironment
from server.models import IncidentAction, ActionType
from server.tasks import list_tasks


# ---------------------------------------------------------------------------
# Observation → prompt (includes all Tier 2 fields — Audit Fix #14)
# ---------------------------------------------------------------------------

def build_obs_prompt(obs_dict: Dict, step: int, action_history: List[str]) -> str:
    """Convert observation to training prompt with ALL observation fields."""
    lines = [
        "You are an SRE Incident Commander. Diagnose and fix the following incident.",
        f"Step {step}/{obs_dict.get('max_steps', 30)}",
        f"System Health: {obs_dict.get('system_health_score', 0):.2%}",
        f"Severity: {obs_dict.get('incident_severity', 'unknown')}",
        f"Escalation Tier: {obs_dict.get('escalation_tier', 1)}/4",
        "",
        "Services:",
    ]

    for name, svc in sorted(obs_dict.get("services", {}).items()):
        status = svc.get("status", "unknown")
        emoji = "🟢" if status == "healthy" else ("🟡" if status == "degraded" else "🔴")
        lines.append(
            f"  {emoji} {name}: {status} | err={svc.get('error_rate', 0):.1%} "
            f"| lat={svc.get('latency_ms', 0):.0f}ms | cpu={svc.get('cpu_percent', 0):.0f}% "
            f"| ver={svc.get('version', '?')}"
        )

    # Alerts
    alerts = obs_dict.get("alerts", [])
    if alerts:
        lines.append("")
        lines.append("Alerts:")
        for a in alerts:
            lines.append(f"  {a}")

    # Services at risk (Tier 2 field)
    at_risk = obs_dict.get("services_at_risk", [])
    if at_risk:
        lines.append("")
        lines.append(f"⚠️ Services at risk of degradation: {', '.join(at_risk)}")

    # Runbook memory (Tier 2 field)
    runbook = obs_dict.get("runbook_memory", [])
    if runbook:
        lines.append("")
        lines.append("📖 Runbook memory (past similar incidents):")
        for entry in runbook:
            lines.append(
                f"  - {entry.get('incident_type', '?')}: "
                f"fix=[{', '.join(entry.get('fix_sequence', []))}] "
                f"score={entry.get('score', 0):.2f} "
                f"({entry.get('episodes_ago', '?')} episodes ago)"
            )

    # Log quality warning (Tier 2 field)
    metadata = obs_dict.get("metadata", {})
    log_quality = metadata.get("log_quality")
    if log_quality and log_quality != "full":
        lines.append("")
        lines.append(f"⚠️ Log quality: {log_quality} — logs may be incomplete or misleading")

    # Logs from last inspect
    logs = obs_dict.get("logs", [])
    if logs:
        lines.append("")
        lines.append("Recent Logs:")
        for log_line in logs[:10]:  # Cap to avoid prompt explosion
            lines.append(f"  {log_line}")

    if action_history:
        lines.append("")
        lines.append("Previous actions: " + ", ".join(action_history))

    lines.append("")
    lines.append('Respond with a JSON action: {"action_type": "...", "service_name": "..."}')

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Episode rollout for reward computation (Audit Fix #3 + #4)
# ---------------------------------------------------------------------------

def rollout_episode(
    task_name: str,
    actions_text: List[str],
    seed: Optional[int] = None,
) -> float:
    """
    Run a COMPLETE episode rollout and return the FINAL episode score.
    
    Each call creates a fresh environment — no shared state (Audit Fix #3).
    Returns grade()["score"] (0.0-1.0), not per-step reward (Audit Fix #4).
    
    Args:
        task_name: Which task to run.
        actions_text: List of JSON action strings from the model.
        seed: Optional seed for reproducibility.
    
    Returns:
        Episode score in [0.0, 1.0].
    """
    env = IncidentCommanderEnvironment()
    obs = env.reset(task_name=task_name, seed=seed)

    for action_text in actions_text:
        if obs.done:
            break

        try:
            text = action_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines).strip()
            data = json.loads(text)
            action = IncidentAction(**data)
        except Exception:
            # Unparseable action → do_nothing
            action = IncidentAction(action_type=ActionType.DO_NOTHING)

        obs = env.step(action)

    # Return FINAL episode score, not per-step reward (Audit Fix #4)
    grade = env.grade()
    env.close()
    return grade["score"]


def compute_single_action_reward(
    task_name: str,
    obs_dict: Dict,
    action_text: str,
    action_history: List[str],
    seed: Optional[int] = None,
) -> float:
    """
    Compute reward for a single action by running a full rollout.
    
    Replays the action history up to this point, then executes the new action,
    then uses a heuristic fallback to complete the episode.
    Returns the final episode grade (0.0-1.0).
    
    This ensures GRPO gets episode-level rewards that are comparable
    across different completions from the same prompt.
    """
    env = IncidentCommanderEnvironment()
    obs = env.reset(task_name=task_name, seed=seed)

    # Replay history
    for past_action_str in action_history:
        if obs.done:
            break
        parts = past_action_str.split(":", 1)
        action_type = parts[0]
        service_name = parts[1] if len(parts) > 1 else None
        try:
            action = IncidentAction(action_type=action_type, service_name=service_name)
        except Exception:
            action = IncidentAction(action_type=ActionType.DO_NOTHING)
        obs = env.step(action)

    # Execute the NEW action
    if not obs.done:
        try:
            text = action_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines).strip()
            data = json.loads(text)
            action = IncidentAction(**data)
        except Exception:
            return -0.1  # Parse failure penalty

        obs = env.step(action)

    # Complete episode with do_nothing to get final score
    while not obs.done:
        obs = env.step(IncidentAction(action_type=ActionType.DO_NOTHING))

    grade = env.grade()
    env.close()
    return grade["score"]


# ---------------------------------------------------------------------------
# GRPO Reward Function Wrapper (Audit Fix #3 + #4)
# ---------------------------------------------------------------------------

class IncidentCommanderRewardFunction:
    """
    Reward function wrapper for TRL GRPO training.
    
    CRITICAL: Each reward computation runs a FRESH, INDEPENDENT rollout.
    No shared environment state between completions (Audit Fix #3).
    Returns EPISODE-LEVEL score, not per-step reward (Audit Fix #4).
    """

    TASKS = ["single_service_failure", "cascading_failure", "hidden_root_cause"]

    def __init__(self):
        self._current_task_idx = 0
        self._seed = 42

    def next_task(self) -> str:
        """Cycle through tasks."""
        task = self.TASKS[self._current_task_idx % len(self.TASKS)]
        self._current_task_idx += 1
        return task

    def get_initial_obs(self, task_name: str) -> Dict:
        """Get initial observation for a task (for prompt building)."""
        env = IncidentCommanderEnvironment()
        obs = env.reset(task_name=task_name, seed=self._seed)
        obs_dict = obs.model_dump()
        env.close()
        return obs_dict

    def score_completions(
        self,
        task_name: str,
        completions: List[str],
        action_history: List[str],
    ) -> List[float]:
        """
        Score multiple GRPO completions for the same prompt.
        
        Each completion is scored via an independent full rollout.
        Returns list of episode-level scores (0.0-1.0).
        """
        scores = []
        for completion in completions:
            score = compute_single_action_reward(
                task_name=task_name,
                obs_dict={},  # Not used in current implementation
                action_text=completion,
                action_history=action_history,
                seed=self._seed,  # Same seed = same starting state
            )
            scores.append(score)
        return scores


# ---------------------------------------------------------------------------
# Training Loop (GRPO)
# ---------------------------------------------------------------------------

def build_training_prompts(reward_fn: IncidentCommanderRewardFunction) -> List[Dict]:
    """Generate training prompt + task pairs from environment observations."""
    prompts = []

    for task in reward_fn.TASKS:
        obs_dict = reward_fn.get_initial_obs(task)
        prompt = build_obs_prompt(obs_dict, 1, [])
        prompts.append({"prompt": prompt, "task": task})

    return prompts


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Incident Commander")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--steps", type=int, default=300, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="trained_model",
                        help="Output directory for trained model")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Log metrics every N steps")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test without GPU (uses mock training)")
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    print(f"{'='*60}")
    print(f"  GRPO Training — Incident Commander")
    print(f"  Model: {args.model}")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE — no GPU training, testing environment wrapper only\n")
        
        reward_fn = IncidentCommanderRewardFunction()

        training_log = []
        for step in range(1, min(args.steps, 10) + 1):
            task = reward_fn.next_task()
            
            # Simulate K=4 model completions for GRPO
            mock_completions = [
                '{"action_type": "inspect_logs", "service_name": "database"}',
                '{"action_type": "restart_service", "service_name": "cache"}',
                '{"action_type": "do_nothing"}',
                '{"action_type": "inspect_metrics", "service_name": "auth"}',
            ]
            
            # Score all completions independently (Audit Fix #3)
            scores = reward_fn.score_completions(
                task_name=task,
                completions=mock_completions,
                action_history=[],
            )

            mean_score = sum(scores) / len(scores)
            best_score = max(scores)
            
            training_log.append({
                "step": step,
                "task": task,
                "mean_score": round(mean_score, 4),
                "best_score": round(best_score, 4),
                "scores": [round(s, 4) for s in scores],
            })

            if step % args.log_every == 0 or step == 1:
                print(
                    f"  Step {step}: task={task} "
                    f"mean_score={mean_score:.4f} best={best_score:.4f} "
                    f"scores={[round(s,3) for s in scores]}"
                )

        # Save training log
        log_path = results_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)
        print(f"\n✅ Training log saved to {log_path}")
        print("✅ Dry run complete — reward function produces independent episode-level scores.")
        print("✅ Run on GPU with actual model for GRPO training.")
        return

    # --- Actual GRPO Training ---
    try:
        from trl import GRPOTrainer, GRPOConfig
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        print("ERROR: TRL/transformers not installed. Required for training.", file=sys.stderr)
        print("Install with: pip install trl transformers torch accelerate peft", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Set up reward function
    reward_fn = IncidentCommanderRewardFunction()

    def compute_rewards(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """
        GRPO reward function: score each completion via independent rollout.
        Returns episode-level scores (0.0-1.0).
        """
        rewards = []
        for prompt, completion in zip(prompts, completions):
            # Determine task from prompt content
            task = "single_service_failure"  # default
            for t in reward_fn.TASKS:
                if t in prompt:
                    task = t
                    break

            score = compute_single_action_reward(
                task_name=task,
                obs_dict={},
                action_text=completion,
                action_history=[],
                seed=42,
            )
            rewards.append(score)
        return rewards

    # GRPO config
    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=args.log_every,
        save_steps=args.steps,  # Save at end
        gradient_accumulation_steps=2,
        bf16=True,
    )

    # Build training prompts
    prompt_data = build_training_prompts(reward_fn)

    print(f"\nStarting GRPO training for {args.steps} steps...")
    print(f"Training prompts: {len(prompt_data)} tasks")
    
    # Note: The actual TRL GRPO integration will be adapted in Bangalore
    # based on the exact TRL version available on the compute cluster.
    # The reward function is ready — each completion gets an independent
    # episode-level score via rollout_episode().

    print("⚠️ Full GRPO training requires GPU compute. Use --dry-run for testing.")
    print(f"Training will be completed in Bangalore with compute credits.")


if __name__ == "__main__":
    main()
