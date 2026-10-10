"""``training/rollout_failure/*`` per-class counts and the dropped-rollout guard.

``AgentCoreAgentLoop`` drops rollouts that failed for non-model reasons (see
``backends/verl/rollout_failure.py``). A dropped rollout leaves no training row, so a
deterministic agent/environment bug can drop most of every step while training carries
on with almost no data. This mixin:

- owns the named Ray actor the agent loops record each failure decision in, and reports
  this step's counts as ``training/rollout_failure/total_<class>_<action>``;
- reports ``training/rollout_failure/drop_fraction`` = ``total_missing_sessions`` /
  (``data.train_batch_size`` * ``rollout.n``);
- raises :class:`RolloutFailureGuardError` once ``drop_fraction`` exceeds
  ``trainer.v1.agentcore_max_drop_fraction`` (default 0.5; ``null`` disables) for
  ``trainer.v1.agentcore_drop_guard_steps`` (default 3) consecutive steps.

Both keys are new to verl's config, so set them with a ``+`` Hydra override, e.g.
``+trainer.v1.agentcore_max_drop_fraction=0.3``. ``total_missing_sessions`` comes from
``VariableRowBatchingMixin``; without it, the guard is inactive.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from omegaconf import DictConfig

from ..rollout_failure import REPORTED_EVENTS, create_stats_actor
from .base import TrainerMixinBase

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

MISSING_SESSIONS_KEY = "training/rollout_failure/total_missing_sessions"
DROP_FRACTION_KEY = "training/rollout_failure/drop_fraction"
_DEFAULT_MAX_DROP_FRACTION = 0.5
_DEFAULT_DROP_GUARD_STEPS = 3
_STATS_TIMEOUT_S = 10.0


class RolloutFailureGuardError(RuntimeError):
    """Too many rollouts were dropped for too many consecutive steps."""


def _guard_config(config: DictConfig) -> tuple[float | None, int]:
    v1 = config.trainer.get("v1") or {}
    fraction = v1.get("agentcore_max_drop_fraction", _DEFAULT_MAX_DROP_FRACTION)
    steps = v1.get("agentcore_drop_guard_steps", _DEFAULT_DROP_GUARD_STEPS)
    if fraction is not None:
        if isinstance(fraction, bool) or not isinstance(fraction, int | float) or fraction < 0:
            raise ValueError(f"trainer.v1.agentcore_max_drop_fraction must be a number >= 0 or null, got {fraction!r}")
        fraction = float(fraction)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError(f"trainer.v1.agentcore_drop_guard_steps must be a positive integer, got {steps!r}")
    return fraction, steps


class RolloutFailureGuardMixin(TrainerMixinBase):
    """Adds ``training/rollout_failure/*`` counts and fails the run on sustained drops."""

    def __init__(self, config: DictConfig):
        self._max_drop_fraction, self._drop_guard_steps = _guard_config(config)
        super().__init__(config)
        self._drop_streak = 0
        self._step_failure_counts: dict[str, int] | None = None
        # Created here so the trainer process owns it for the whole run; the agent
        # loops find it by name. None without a Ray cluster (counts then unavailable).
        self._failure_stats = create_stats_actor()

    def on_step_end(self):
        super().on_step_end()
        # Taken before validation runs: validation rollouts carry the same global_steps.
        self._step_failure_counts = self._pop_failure_counts(self.global_steps)

    def _pop_failure_counts(self, step: int) -> dict[str, int] | None:
        if self._failure_stats is None:
            return None
        try:
            import ray

            return ray.get(self._failure_stats.pop.remote(step), timeout=_STATS_TIMEOUT_S)
        except Exception as e:
            logger.warning("could not read rollout-failure counts for step %s: %s", step, e)
            return None

    def _compute_metrics(
        self,
        batch: Any,
        metrics: dict[str, Any],
        timing_raw: dict[str, Any],
        global_steps: int,
        epoch: int,
    ) -> None:
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)

        counts, self._step_failure_counts = self._step_failure_counts, None
        if counts is not None:
            for failure_class, action in REPORTED_EVENTS:
                key = f"training/rollout_failure/total_{failure_class}_{action}"
                metrics[key] = counts.get(f"{failure_class}/{action}", 0)

        missing = metrics.get(MISSING_SESSIONS_KEY)
        if missing is None:
            return
        nominal = int(self.config.data.train_batch_size) * int(self.config.actor_rollout_ref.rollout.n)
        fraction = float(missing) / nominal if nominal > 0 else 0.0
        metrics[DROP_FRACTION_KEY] = fraction
        self._check_drop_guard(fraction, missing, nominal, counts, metrics, global_steps)

    def _check_drop_guard(
        self,
        fraction: float,
        missing: Any,
        nominal: int,
        counts: dict[str, int] | None,
        metrics: dict[str, Any],
        global_steps: int,
    ) -> None:
        if self._max_drop_fraction is None:
            return
        self._drop_streak = self._drop_streak + 1 if fraction > self._max_drop_fraction else 0
        if self._drop_streak < self._drop_guard_steps:
            return

        per_class = (
            ", ".join(f"{k}={v}" for k, v in sorted(counts.items()) if v) or "none recorded"
            if counts is not None
            else "unavailable (no stats actor)"
        )
        message = (
            f"{self._drop_streak} consecutive steps dropped more than "
            f"{self._max_drop_fraction:.0%} of rollouts (trainer.v1.agentcore_max_drop_fraction). "
            f"Step {global_steps}: {missing}/{nominal} rollouts missing (drop_fraction={fraction:.3f}); "
            f"failure counts this step: {per_class}. A deterministic agent or environment bug is the "
            "likely cause; see the [rollout-failure] lines in the driver log."
        )
        # The trainer loop logs a step's metrics after _compute_metrics, which never
        # happens once this raises; log them here so the failing step stays visible.
        try:
            scalars = {k: v for k, v in metrics.items() if isinstance(v, int | float)}
            self.logger.log(data=scalars, step=global_steps)
        except Exception:
            logger.warning("could not log the final step's metrics before failing", exc_info=True)
        raise RolloutFailureGuardError(message)
