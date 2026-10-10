"""Rollout-failure policy pieces outside the agent loop: the classification table, the
per-step stats recording, and ``RolloutFailureGuardMixin``'s metrics and drop guard.

The trainer is a stub exposing only what the mixin reads, and the Ray stats actor is a
fake handle (no cluster in unit tests).
"""

import pytest
from omegaconf import OmegaConf

from agentcore_rl_toolkit.backends.verl import rollout_failure as rf
from agentcore_rl_toolkit.backends.verl.trainer_mixins import RolloutFailureGuardError, RolloutFailureGuardMixin
from agentcore_rl_toolkit.backends.verl.trainer_mixins.rollout_failure_guard import (
    DROP_FRACTION_KEY,
    MISSING_SESSIONS_KEY,
)

# -- classification -------------------------------------------------------------


def _classify(**overrides):
    args = {
        "phase": None,
        "status_code": 500,
        "num_turns": 2,
        "context_exhausted": False,
        "last_finish_reason": "stop",
        "timeout_policy": "drop",
    }
    return rf.classify_failure(**{**args, **overrides})


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"context_exhausted": True}, "model"),
        ({"last_finish_reason": "length"}, "model"),
        ({"context_exhausted": True, "phase": "timeout"}, "model"),  # model rules match first
        ({"phase": "timeout"}, "timeout"),
        ({"phase": "timeout", "timeout_policy": "penalize"}, "model"),
        ({"phase": "invoke", "status_code": None, "num_turns": 0}, "transient"),
        ({"phase": "poll", "status_code": None}, "transient"),
        ({"num_turns": 0}, "transient"),  # 500 before any model turn
        ({}, "agent_error"),  # 500 after model turns
        ({"status_code": 400, "num_turns": 0}, "agent_error"),
    ],
)
def test_classification_table(overrides, expected):
    assert _classify(**overrides) == expected


def test_rollout_dropped_is_a_runtime_error_carrying_its_class():
    err = rf.RolloutDropped("agent_error", "boom")
    assert isinstance(err, RuntimeError)
    assert err.failure_class == "agent_error"
    assert "class=agent_error" in str(err)


# -- stats ----------------------------------------------------------------------


def test_stats_pop_returns_the_step_and_discards_older_steps():
    stats = rf.RolloutFailureStats()
    stats.record(1, "agent_error", "drop")  # e.g. a late validation rollout of step 1
    stats.record(2, "agent_error", "drop")
    stats.record(2, "agent_error", "drop")
    stats.record(2, "transient", "retry")
    stats.record(3, "timeout", "drop")
    assert stats.pop(2) == {"agent_error/drop": 2, "transient/retry": 1}
    assert stats.pop(1) == {}  # discarded
    assert stats.pop(3) == {"timeout/drop": 1}


@pytest.mark.asyncio
async def test_record_failure_without_ray_is_a_noop():
    await rf.record_failure(1, "timeout", "drop")  # must not raise


class _FakeRemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args):
        async def call():  # an awaitable, like a Ray ObjectRef
            return self._fn(*args)

        return call()


class _FakeStatsActor:
    """Stands in for the named Ray actor handle (actor methods via ``.remote``)."""

    def __init__(self, fail: bool = False):
        self.stats = rf.RolloutFailureStats()
        self.record = _FakeRemoteMethod(self._fail if fail else self.stats.record)

    @staticmethod
    def _fail(*args):
        raise RuntimeError("actor died")


@pytest.mark.asyncio
async def test_record_failure_awaits_the_stats_actor(monkeypatch):
    actor = _FakeStatsActor()
    monkeypatch.setattr(rf, "_stats_actor", actor)
    await rf.record_failure(4, "agent_error", "drop")
    await rf.record_failure(4, "transient", "retry")
    assert actor.stats.pop(4) == {"agent_error/drop": 1, "transient/retry": 1}


@pytest.mark.asyncio
async def test_record_failure_never_raises_and_forgets_a_dead_actor(monkeypatch):
    monkeypatch.setattr(rf, "_stats_actor", _FakeStatsActor(fail=True))
    await rf.record_failure(4, "timeout", "drop")
    assert rf._stats_actor is None  # looked up again on the next failure


# -- guard mixin ----------------------------------------------------------------


class _StubTrainer:
    def __init__(self, config):
        self.config = config
        self.global_steps = 1
        self.logged: list[tuple[int, dict]] = []
        self.logger = self

    def log(self, data, step):
        self.logged.append((step, data))

    def on_step_end(self):
        pass

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        pass


class _Trainer(RolloutFailureGuardMixin, _StubTrainer):
    pass


def _make_trainer(v1: dict | None = None, *, counts=None) -> _Trainer:
    config = OmegaConf.create(
        {
            "trainer": {"v1": {"trainer_mode": "sync", **(v1 or {})}},
            "data": {"train_batch_size": 4},
            "actor_rollout_ref": {"rollout": {"n": 4}},  # 16 nominal rollouts per step
        }
    )
    OmegaConf.set_struct(config, True)  # verl's config is struct: unknown keys must read as defaults
    trainer = _Trainer(config)
    trainer._pop_failure_counts = lambda step: counts
    return trainer


def _step(trainer: _Trainer, missing: int | None) -> dict:
    trainer.on_step_end()
    metrics = {} if missing is None else {MISSING_SESSIONS_KEY: missing}
    trainer._compute_metrics(None, metrics, {}, trainer.global_steps, 0)
    trainer.global_steps += 1
    return metrics


def test_three_consecutive_steps_over_the_threshold_raise_with_counts():
    trainer = _make_trainer(counts={"agent_error/drop": 9, "transient/retry": 1})
    assert _step(trainer, 9)[DROP_FRACTION_KEY] == pytest.approx(9 / 16)
    _step(trainer, 12)
    with pytest.raises(RolloutFailureGuardError) as excinfo:
        _step(trainer, 16)
    message = str(excinfo.value)
    assert "3 consecutive steps" in message
    assert "16/16 rollouts missing" in message
    assert "agent_error/drop=9" in message and "transient/retry=1" in message
    # the failing step's metrics are still logged
    assert trainer.logged[-1][0] == 3
    assert trainer.logged[-1][1]["training/rollout_failure/total_agent_error_drop"] == 9


def test_non_consecutive_steps_over_the_threshold_do_not_raise():
    trainer = _make_trainer()
    for missing in [9, 12, 8, 16, 16, 2, 9, 9]:  # 8/16 is not *more* than 0.5
        _step(trainer, missing)


def test_threshold_and_patience_are_configurable():
    trainer = _make_trainer({"agentcore_max_drop_fraction": 0.1, "agentcore_drop_guard_steps": 1})
    with pytest.raises(RolloutFailureGuardError, match="1 consecutive steps"):
        _step(trainer, 2)


def test_null_threshold_disables_the_guard():
    trainer = _make_trainer({"agentcore_max_drop_fraction": None})
    for _ in range(5):
        _step(trainer, 16)


@pytest.mark.parametrize(
    "v1",
    [
        {"agentcore_max_drop_fraction": -0.1},
        {"agentcore_max_drop_fraction": "half"},
        {"agentcore_drop_guard_steps": 0},
        {"agentcore_drop_guard_steps": True},
    ],
)
def test_invalid_guard_config_rejected(v1):
    with pytest.raises(ValueError, match="agentcore_"):
        _make_trainer(v1)


def test_per_class_counts_are_reported_every_step():
    trainer = _make_trainer(counts={"model/train": 2})
    metrics = _step(trainer, 0)
    assert metrics["training/rollout_failure/total_model_train"] == 2
    for failure_class, action in rf.REPORTED_EVENTS:
        assert f"training/rollout_failure/total_{failure_class}_{action}" in metrics


def test_guard_inactive_without_missing_sessions_metric():
    trainer = _make_trainer({"agentcore_drop_guard_steps": 1})
    assert DROP_FRACTION_KEY not in _step(trainer, None)


def test_registered_sync_trainer_includes_the_guard():
    from agentcore_rl_toolkit.backends.verl.trainer import (
        AgentCorePPOTrainerColocateAsync,
        AgentCorePPOTrainerSync,
    )

    assert issubclass(AgentCorePPOTrainerSync, RolloutFailureGuardMixin)
    assert not issubclass(AgentCorePPOTrainerColocateAsync, RolloutFailureGuardMixin)
