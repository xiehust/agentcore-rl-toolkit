"""Rollout failure policy shared by ``AgentCoreAgentLoop`` and the trainer guard.

A failed rollout is classified, then trained at reward 0, retried, or dropped:

- ``model``: the model caused it (the gateway answered a context-limit error for the
  session, the last turn stopped on ``finish_reason="length"``, or a timeout under
  ``timeout_policy="penalize"``). Trained at reward 0, as before.
- ``transient``: infrastructure (``invoke_async`` raised, S3 polling raised, or the agent
  returned ``status_code=500`` without a single captured model turn). Retried on a fresh
  session id, then dropped.
- ``agent_error``: any other non-200 result (agent or environment code raised after
  model turns). Dropped.
- ``timeout``: the shared ``max_rollout_time`` deadline passed. Dropped.

Dropping raises :class:`RolloutDropped`; verl's sync trainer drops only that trajectory and
trains its siblings. Every decision is logged as one ``[rollout-failure]`` line and, when
the AgentCore trainer runs, counted in a Ray actor the trainer reads once per step
(Ray's driver log deduplication can collapse identical-looking worker lines, so the
metrics are the reliable count).

Must stay importable without a Ray cluster: ``ray`` is imported lazily.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

FAILURE_CLASSES = ("model", "transient", "agent_error", "timeout")
FAILURE_ACTIONS = ("train", "retry", "drop")
TIMEOUT_POLICIES = ("drop", "penalize")

# Every (class, action) pair the agent loop can emit; the trainer reports each one
# every step (0 when absent) so the metric series stay continuous.
REPORTED_EVENTS = (
    ("model", "train"),
    ("model", "drop"),
    ("transient", "retry"),
    ("transient", "drop"),
    ("agent_error", "train"),
    ("agent_error", "drop"),
    ("timeout", "drop"),
)

_REASON_MAX_CHARS = 200
STATS_ACTOR_NAME = "agentcore_rollout_failure_stats"
_RECORD_TIMEOUT_S = 5.0


class RolloutDropped(RuntimeError):
    """A failed rollout that must not train. ``failure_class`` is one of
    ``FAILURE_CLASSES``."""

    def __init__(self, failure_class: str, message: str):
        super().__init__(f"rollout dropped (class={failure_class}): {message}")
        self.failure_class = failure_class


def classify_failure(
    *,
    phase: str | None,
    status_code: Any,
    num_turns: int,
    context_exhausted: bool,
    last_finish_reason: str | None,
    timeout_policy: str,
) -> str:
    """Classify a failed rollout attempt, matching the rules in order.

    ``phase`` is where the attempt failed: ``"invoke"`` (``invoke_async`` raised),
    ``"poll"`` (waiting for the S3 result raised), ``"timeout"``, or ``None`` when a
    result arrived (the failure is then its non-200 ``status_code``).
    """
    if context_exhausted or last_finish_reason == "length":
        return "model"
    if phase == "timeout":
        return "model" if timeout_policy == "penalize" else "timeout"
    if phase in ("invoke", "poll"):
        return "transient"
    if status_code == 500 and num_turns == 0:
        return "transient"
    return "agent_error"


def format_reason(reason: str) -> str:
    """The log contract's reason: newlines replaced by spaces, then cut to 200 chars."""
    return reason.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")[:_REASON_MAX_CHARS]


def log_failure(failure_class: str, action: str, sid: str, step: int, reason: str) -> None:
    """Emit the one-line ``[rollout-failure]`` contract parsed by downstream tooling."""
    logger.warning(
        "[rollout-failure] class=%s action=%s sid=%s step=%d reason=%s",
        failure_class,
        action,
        sid,
        step,
        format_reason(reason),
    )


# -- cross-process counts ------------------------------------------------------------
#
# Dropped rollouts leave no row in the training batch, so the trainer cannot count them
# by class from the batch. The trainer creates (and owns) this named actor; agent-loop
# workers look it up by name and record each decision. Without it (plain verl trainer,
# unit tests) recording is a no-op. Recording never fails a rollout.


class RolloutFailureStats:
    """Per-step ``{"<class>/<action>": count}``. Wrapped with ``ray.remote`` by
    :func:`create_stats_actor`; plain class so it is testable without Ray."""

    def __init__(self) -> None:
        self._counts: dict[int, dict[str, int]] = {}

    def record(self, step: int, failure_class: str, action: str) -> None:
        key = f"{failure_class}/{action}"
        per_step = self._counts.setdefault(int(step), {})
        per_step[key] = per_step.get(key, 0) + 1

    def pop(self, step: int) -> dict[str, int]:
        """This step's counts. Older steps are discarded (e.g. validation rollouts
        recorded after the step's counts were already taken)."""
        step = int(step)
        counts = self._counts.pop(step, {})
        for stale in [s for s in self._counts if s < step]:
            del self._counts[stale]
        return counts


def create_stats_actor() -> Any | None:
    """Create the named stats actor, owned by the calling (trainer) process.
    Returns its handle, or ``None`` when Ray is not initialized or creation fails."""
    try:
        import ray

        if not ray.is_initialized():
            return None
        return ray.remote(RolloutFailureStats).options(name=STATS_ACTOR_NAME, get_if_exists=True, num_cpus=0).remote()
    except Exception:
        logger.warning("could not create the rollout-failure stats actor; per-class counts disabled", exc_info=True)
        return None


_stats_actor: Any | None = None


def _lookup_stats_actor() -> Any | None:
    global _stats_actor
    if _stats_actor is not None:
        return _stats_actor
    try:
        import ray

        if not ray.is_initialized():
            return None
        _stats_actor = ray.get_actor(STATS_ACTOR_NAME)
    except Exception:
        return None  # no AgentCore trainer in this job; retried on the next failure
    return _stats_actor


async def record_failure(step: int, failure_class: str, action: str) -> None:
    """Count one decision in the trainer's stats actor; best-effort."""
    global _stats_actor
    actor = _lookup_stats_actor()
    if actor is None:
        return
    try:
        # Awaited (not fire-and-forget) so the count lands before the rollout returns,
        # i.e. before the trainer reads the step's counts.
        await asyncio.wait_for(actor.record.remote(step, failure_class, action), _RECORD_TIMEOUT_S)
    except Exception:
        _stats_actor = None  # actor gone or unreachable: look it up again next time
        logger.debug("rollout-failure stats record failed", exc_info=True)


__all__ = [
    "FAILURE_ACTIONS",
    "FAILURE_CLASSES",
    "REPORTED_EVENTS",
    "STATS_ACTOR_NAME",
    "TIMEOUT_POLICIES",
    "RolloutDropped",
    "RolloutFailureStats",
    "classify_failure",
    "create_stats_actor",
    "format_reason",
    "log_failure",
    "record_failure",
]
