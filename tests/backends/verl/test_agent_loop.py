"""AgentCoreAgentLoop tests: conversion math, invoke wiring, rewards, and failure
paths. The loop is constructed through verl's own ``AgentLoopBase`` against a live
(threaded) gateway; only the LLM server client and the RolloutClient (no AWS) are
faked."""

import logging
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from agentcore_rl_toolkit.backends.verl import agent_loop as al
from agentcore_rl_toolkit.backends.verl.agent_loop import AgentCoreAgentLoop
from agentcore_rl_toolkit.rollout_gateway import TraceRecord

from .conftest import FakeLLMServerClient, FakeTokenizer, make_data_config, make_trainer_config

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """No real backoff sleeps between transient retries."""
    monkeypatch.setattr(al, "_RETRY_BACKOFF_S", (0.0, 0.0))


def _make_loop(llm_client=None, *, use_v1=True, trainer_config=None, **loop_kwargs):
    loop_kwargs.setdefault("max_tokens_per_turn", 8)
    with patch("agentcore_rl_toolkit.backends.verl.agent_loop.RolloutClient") as client_cls:
        client_cls.return_value = MagicMock()
        loop = AgentCoreAgentLoop(
            trainer_config or make_trainer_config(use_v1=use_v1),
            llm_client or FakeLLMServerClient(),
            FakeTokenizer(),
            None,
            None,
            make_data_config(),
            agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/test",
            s3_bucket="test-bucket",
            gateway_bind_host="127.0.0.1",
            gateway_public_host="127.0.0.1",
            name="agentcore_agent",  # hydra passes the YAML entry's name through
            **loop_kwargs,
        )
    return loop


def _wire_result(loop, result: dict[str, Any], *, drive_turns: int = 1):
    """Make invoke_async return a future whose result_async first drives
    ``drive_turns`` chat turns against the live gateway (simulating the ACR
    agent calling back in), then returns ``result``."""

    async def fake_invoke_async(payload, session_id=None, input_id=None, **overrides):
        future = MagicMock()

        async def result_async(timeout=None):
            async with aiohttp.ClientSession() as http:
                messages = [{"role": "user", "content": "hi"}]
                for _ in range(drive_turns):
                    resp = await http.post(
                        f"{loop._gateway.base_url}/v1/chat/completions",
                        json={"model": "m", "messages": messages},
                        headers={"Authorization": f"Bearer {session_id}"},
                    )
                    assert resp.status == 200
                    body = await resp.json()
                    messages = messages + [body["choices"][0]["message"], {"role": "user", "content": "more"}]
            return result

        future.result_async = result_async
        call = {"payload": payload, "session_id": session_id, "input_id": input_id, **overrides}
        fake_invoke_async.calls.append(call)
        return future

    fake_invoke_async.calls = []
    loop._client.invoke_async = fake_invoke_async
    return fake_invoke_async


async def test_run_end_to_end_inline_reward():
    llm = FakeLLMServerClient()
    loop = _make_loop(llm)
    invoke = _wire_result(loop, {"status_code": 200, "rewards": 0.75})

    outputs = await loop.run(
        {"temperature": 0.7},
        raw_prompt=[{"role": "user", "content": "hi"}],
        payload={"question": "2+2?"},
        uid="u1",
    )

    assert len(outputs) == 1
    out = outputs[0]
    # response region = the two generated tokens from FakeLLMServerClient
    assert out.response_ids == [101, 102]
    assert out.response_mask == [1, 1]
    assert out.response_logprobs == [-0.5, -0.6]
    assert len(out.prompt_ids) > 0
    assert out.reward_score == 0.75
    assert out.extra_fields["acr_result"]["rewards"] == 0.75
    assert out.extra_fields["reward_extra_info"] == {
        "agent_reward": 0.75,
        "acr_failed": 0.0,
        "num_trace_records": 1.0,
    }

    # invoke wiring: sid is a 36-char uuid used as both ACR session id and Bearer sid
    call = invoke.calls[0]
    assert len(call["session_id"]) == 36
    # this loop hands the agent its gateway session key via _rollout.api_key
    assert call["api_key"] == call["session_id"]
    # OpenAI-SDK convention: the advertised base_url carries the /v1 prefix
    assert call["base_url"] == f"{loop._gateway.base_url}/v1"
    assert call["model_id"] == "test/model"
    assert call["input_id"] == "u1"
    # the payload column is forwarded verbatim; row plumbing never leaks in
    assert call["payload"] == {"question": "2+2?"}
    # LLM client got the sid as sticky request_id
    assert llm.calls[0]["request_id"] == call["session_id"]


async def test_built_in_mode_missing_reward_warns_and_scores_zero(caplog):
    loop = _make_loop()  # default reward_mode="built_in"
    _wire_result(loop, {"status_code": 200, "artifacts": {"x": 1}})
    with caplog.at_level(logging.WARNING):
        outputs = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")
    assert outputs[0].reward_score == 0.0
    assert any("returned no {'rewards'" in r.message for r in caplog.records)


async def test_malformed_reward_raises():
    """A non-numeric reward means broken agent-side reward code, which would be
    broken on every rollout — raising keeps the unscored row out of the batch
    instead of silently zeroing it and flattening the GRPO group."""
    loop = _make_loop()
    _wire_result(loop, {"status_code": 200, "rewards": "invalid"})
    with pytest.raises(ValueError, match="non-numeric built-in reward"):
        await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")


async def test_malformed_reward_on_failed_rollout_still_scores_zero():
    """With drop_agent_errors=False, a failed rollout with a real partial trace
    trains that trace at reward 0."""
    loop = _make_loop(drop_agent_errors=False)
    _wire_result(loop, {"status_code": 500, "stop_reason": "boom", "rewards": "invalid"})
    outputs = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")
    assert outputs[0].reward_score == 0.0


async def test_empty_reward_list_treated_as_missing(caplog):
    loop = _make_loop()
    _wire_result(loop, {"status_code": 200, "rewards": []})
    with caplog.at_level(logging.WARNING):
        outputs = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")
    assert outputs[0].reward_score == 0.0
    assert any("returned no {'rewards'" in r.message for r in caplog.records)


async def test_separate_reward_mode_rejected():
    """Trainer-side scoring needs data_source/reward_model.ground_truth columns
    the payload-first dataset contract doesn't provide; rejected at construction
    rather than KeyError-ing inside verl's reward manager."""
    with pytest.raises(ValueError, match="reward_mode='separate' is not supported"):
        _make_loop(reward_mode="separate")


async def test_invalid_reward_mode_rejected():
    with pytest.raises(ValueError, match="reward_mode"):
        with patch("agentcore_rl_toolkit.backends.verl.agent_loop.RolloutClient"):
            AgentCoreAgentLoop(
                make_trainer_config(),
                FakeLLMServerClient(),
                FakeTokenizer(),
                None,
                None,
                make_data_config(),
                agent_runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/test",
                s3_bucket="test-bucket",
                gateway_bind_host="127.0.0.1",
                gateway_public_host="127.0.0.1",
                reward_mode="nope",
                max_tokens_per_turn=8,
            )


async def test_multi_turn_merges_to_one_record():
    loop = _make_loop()
    _wire_result(loop, {"status_code": 200, "rewards": 1.0}, drive_turns=3)
    outputs = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")
    # CLEAN prefix extensions merge into ONE record with 3 trained turns
    assert len(outputs) == 1
    assert sum(outputs[0].response_mask) == 6  # 3 turns x 2 generated tokens
    assert outputs[0].num_turns == 4  # 3 LLM turns + 1


async def test_timeout_without_trace_raises():
    loop = _make_loop()

    async def fake_invoke_async(payload, session_id=None, input_id=None, **overrides):
        future = MagicMock()
        future.result_async = AsyncMock(side_effect=TimeoutError())
        return future

    loop._client.invoke_async = fake_invoke_async
    with pytest.raises(RuntimeError, match="timed out"):
        await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")


async def test_invoke_error_without_trace_raises():
    loop = _make_loop()
    loop._client.invoke_async = AsyncMock(side_effect=RuntimeError("throttled"))
    with pytest.raises(RuntimeError, match="throttled"):
        await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")


async def test_agent_error_status_with_trace_scores_zero_when_not_dropping():
    """drop_agent_errors=False keeps the old behaviour: the agent errored after some
    LLM turns, so the trace trains with the reward forced to 0.0."""
    loop = _make_loop(drop_agent_errors=False)
    _wire_result(loop, {"status_code": 500, "stop_reason": "boom", "rewards": 0.9})
    outputs = await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")
    assert len(outputs) == 1
    assert outputs[0].reward_score == 0.0  # inline reward ignored on failure
    assert "status_code=500" in outputs[0].extra_fields["acr_error"]
    assert outputs[0].extra_fields["rollout_failure_class"] == "agent_error"
    assert sum(outputs[0].response_mask) == 2  # partial trace still trains


async def test_use_v1_required():
    with pytest.raises(ValueError, match="use_v1"):
        _make_loop(use_v1=False)


async def test_static_session_warning(caplog):
    """Stale agent image: the agent calls in with api_key='EMPTY' instead of its
    ACR session id, so the real sid drains empty and raises after warning about
    the static session."""
    loop = _make_loop()

    async def fake_invoke_async(payload, session_id=None, input_id=None, **overrides):
        future = MagicMock()

        async def result_async(timeout=None):
            async with aiohttp.ClientSession() as http:
                resp = await http.post(
                    f"{loop._gateway.base_url}/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer EMPTY"},
                )
                assert resp.status == 200
            return {"status_code": 200, "rewards": 1.0}

        future.result_async = result_async
        return future

    loop._client.invoke_async = fake_invoke_async
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError, match="no trainable trajectory"):
            await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"}, uid="u1")

    assert any("static session 'EMPTY'" in r.message for r in caplog.records)


async def test_exp_id_derived_from_trainer_config():
    loop = _make_loop()
    # conftest's make_trainer_config has no project/experiment name -> defaults
    assert loop._exp_id == "verl-run"


async def test_client_cached_by_config():
    loop1 = _make_loop()
    n_after_first = len(al._CLIENTS)
    loop2 = _make_loop()  # same config -> same cached client
    assert len(al._CLIENTS) == n_after_first
    assert loop1._client is loop2._client


async def test_no_payload_column_raises():
    """A missing `payload` column is a config error, raised loudly (not degraded
    per-rollout): the payload column is the single dataset contract — plain row
    fields are never forwarded (the row namespace is shared with verl plumbing,
    and verl-shaped column values are not agent-shaped)."""
    loop = _make_loop()
    with pytest.raises(ValueError, match="payload"):
        await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], question="2+2?", uid="u1")


async def test_payload_column_forwarded_verbatim():
    """The `payload` dataset column is the agent's exact invoke payload —
    forwarded as-is; sibling row fields never leak in."""
    loop = _make_loop()
    invoke = _wire_result(loop, {"status_code": 200, "rewards": 1.0})
    await loop.run(
        {},
        raw_prompt=[{"role": "user", "content": "hi"}],
        payload={"prompt": "What is 2+2?", "answer": "4"},
        question="ignored",
        uid="u1",
    )
    assert invoke.calls[0]["payload"] == {"prompt": "What is 2+2?", "answer": "4"}


async def test_output_conversion_preserves_prompt_overflow():
    loop = _make_loop(
        trainer_config=make_trainer_config(prompt_length=2, response_length=6, max_model_len=6),
        max_tokens_per_turn=4,
    )
    record = TraceRecord(
        token_ids=[1, 2, 3, 4, 5, 6],
        loss_mask=[1, 1],
        logprobs=[-0.5, -0.6],
    )

    out = loop._record_to_output(record, 0, 1.0, 1, {}, 0.1)

    assert out.prompt_ids == [1, 2]
    assert out.response_ids == [3, 4, 5, 6]
    assert out.response_mask == [0, 0, 1, 1]
    assert out.response_logprobs == [0.0, 0.0, -0.5, -0.6]


async def test_session_budget_matches_response_storage():
    llm = FakeLLMServerClient()
    loop = _make_loop(
        llm,
        trainer_config=make_trainer_config(prompt_length=8, response_length=24, max_model_len=32),
    )
    _wire_result(loop, {"status_code": 200, "rewards": 1.0})
    gateway = loop._gateway.gateway
    with patch.object(gateway, "create_session", wraps=gateway.create_session) as create_session:
        await loop.run({}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"prompt": "hi"})

    max_context_tokens = create_session.call_args.kwargs["max_context_tokens"]
    assert max_context_tokens == 24
    assert max_context_tokens == loop.response_length
    assert max_context_tokens != loop.prompt_length + loop.response_length


async def test_sampling_defaults_passed_to_gateway():
    llm = FakeLLMServerClient()
    loop = _make_loop(llm)
    _wire_result(loop, {"status_code": 200, "rewards": 1.0})
    await loop.run(
        {"temperature": 0.3, "top_p": 0.8, "top_k": 20},
        raw_prompt=[{"role": "user", "content": "hi"}],
        payload={"prompt": "hi"},
    )
    sp = llm.calls[0]["sampling_params"]
    assert sp["temperature"] == 0.3
    assert sp["top_p"] == 0.8
    assert sp["top_k"] == 20
    assert sp["max_new_tokens"] == 8


async def test_explicit_max_model_len_is_required():
    with pytest.raises(ValueError, match="explicit positive.*max_model_len"):
        _make_loop(trainer_config=make_trainer_config(max_model_len=None))


@pytest.mark.parametrize("field", ["prompt_length", "response_length"])
async def test_padded_region_cannot_exceed_max_model_len(field):
    lengths = {field: 129}
    with pytest.raises(ValueError, match=rf"{field} \(129\) cannot exceed .*max_model_len \(128\)"):
        _make_loop(trainer_config=make_trainer_config(max_model_len=128, **lengths))


@pytest.mark.parametrize("value", [0, -1, True, 129])
async def test_invalid_max_tokens_per_turn_rejected(value):
    with pytest.raises(ValueError, match="max_tokens_per_turn"):
        _make_loop(max_tokens_per_turn=value)


async def test_reward_extra_info_carries_scalar_metrics_over_defaults():
    loop = _make_loop(
        FakeLLMServerClient(), reward_extra_info_defaults={"f_beta": 0.0, "num_turns": 0.0, "reward": 0.0}
    )
    _wire_result(
        loop,
        {
            "status_code": 200,
            "rewards": 0.75,
            "metrics": {"f_beta": 0.6, "num_turns": 3, "undeclared": 9, "note": "x"},
        },
    )
    outs = await loop.run(
        {"temperature": 1.0}, raw_prompt=[{"role": "user", "content": "hi"}], payload={"q": 1}, uid="u"
    )
    rei = outs[-1].extra_fields["reward_extra_info"]
    assert rei["f_beta"] == 0.6 and rei["num_turns"] == 3.0
    assert rei["acr_failed"] == 0.0 and rei["num_trace_records"] == 1.0
    assert "reward" not in rei  # verl derives its reward metric from rm_scores
    # ...but the agent's own scalar is carried under a non-colliding key, so it does not
    # append to the list verl fills from rm_scores.
    assert rei["agent_reward"] == 0.75
    assert "undeclared" not in rei and "note" not in rei


async def test_failed_rollout_carries_reward_extra_info_defaults():
    loop = _make_loop(FakeLLMServerClient(), reward_extra_info_defaults={"f_beta": 0.0})
    rei = loop._reward_extra_info(None, 0.0, 0, failed=True)
    assert rei == {"f_beta": 0.0, "agent_reward": 0.0, "acr_failed": 1.0, "num_trace_records": 0.0}


async def test_reward_extra_info_keys_are_stable_across_loop_instances():
    defaults = {"submitted": 0.0, "num_turns": 0.0, "ok": 0.0}
    successful_loop = _make_loop(FakeLLMServerClient(), reward_extra_info_defaults=defaults)
    failed_loop = _make_loop(FakeLLMServerClient(), reward_extra_info_defaults=defaults)

    successful = successful_loop._reward_extra_info(
        {"metrics": {"submitted": 1.0, "num_turns": 4, "ok": True}}, 0.75, 1, failed=False
    )
    failed = failed_loop._reward_extra_info(None, 0.0, 0, failed=True)

    assert successful == {
        "submitted": 1.0,
        "num_turns": 4.0,
        "ok": 1.0,
        "agent_reward": 0.75,
        "acr_failed": 0.0,
        "num_trace_records": 1.0,
    }
    assert failed == {
        "submitted": 0.0,
        "num_turns": 0.0,
        "ok": 0.0,
        "agent_reward": 0.0,
        "acr_failed": 1.0,
        "num_trace_records": 0.0,
    }
    assert set(successful) == set(failed)


# -- failure policy -------------------------------------------------------------
#
# Each attempt spec drives one invoke: {"invoke_error": exc} raises from invoke_async;
# otherwise result_async drives "turns" chat turns (or the "messages_per_turn" list of
# message lists), then raises "poll_error"/"poll_timeout" or returns "result".

_FAILURE_LINE = re.compile(
    r"^\[rollout-failure\] class=(?P<cls>model|transient|agent_error|timeout) "
    r"action=(?P<action>train|retry|drop) sid=(?P<sid>[0-9a-f-]{36}) step=(?P<step>\d+) reason=(?P<reason>.*)$"
)
_ROW = {"raw_prompt": [{"role": "user", "content": "hi"}], "payload": {"prompt": "hi"}, "uid": "u1", "global_steps": 7}


async def _drive(loop, sid: str, message_lists: list[list[dict]]) -> list[int]:
    statuses = []
    async with aiohttp.ClientSession() as http:
        for messages in message_lists:
            resp = await http.post(
                f"{loop._gateway.base_url}/v1/chat/completions",
                json={"model": "m", "messages": messages},
                headers={"Authorization": f"Bearer {sid}"},
            )
            statuses.append(resp.status)
    return statuses


def _wire_attempts(loop, attempts: list[dict]):
    attempts = list(attempts)

    async def fake_invoke_async(payload, session_id=None, input_id=None, **overrides):
        spec = attempts.pop(0)
        fake_invoke_async.calls.append({"session_id": session_id, **overrides})
        if "invoke_error" in spec:
            raise spec["invoke_error"]
        future = MagicMock()
        future.cancel_async = AsyncMock()

        async def result_async(timeout=None):
            fake_invoke_async.timeouts.append(timeout)
            message_lists = spec.get("messages_per_turn")
            if message_lists is None:
                message_lists = [[{"role": "user", "content": "hi"}]] * spec.get("turns", 0)
            fake_invoke_async.statuses.append(await _drive(loop, session_id, message_lists))
            if "poll_error" in spec:
                raise spec["poll_error"]
            if spec.get("poll_timeout"):
                raise TimeoutError()
            return spec["result"]

        future.result_async = result_async
        fake_invoke_async.futures.append(future)
        return future

    fake_invoke_async.calls = []
    fake_invoke_async.timeouts = []
    fake_invoke_async.statuses = []
    fake_invoke_async.futures = []
    loop._client.invoke_async = fake_invoke_async
    return fake_invoke_async


@pytest.fixture
def recorded(monkeypatch):
    """Captures the per-step counts the loop reports to the trainer's stats actor."""
    events: list[tuple[int, str, str]] = []

    async def record(step, failure_class, action):
        events.append((step, failure_class, action))

    monkeypatch.setattr(al, "record_failure", record)
    return events


def _failure_lines(caplog) -> list[dict]:
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[rollout-failure]")]
    parsed = [_FAILURE_LINE.match(line) for line in lines]
    assert all(parsed), lines  # every line matches the log contract exactly
    return [m.groupdict() for m in parsed]


_OK = {"status_code": 200, "rewards": 1.0}
# What @rollout_entrypoint saves when the agent's own code raises (e.g. OfficeBench's
# IndexError on empty content), after the model already took turns.
_INDEX_ERROR = {"status_code": 500, "stop_reason": "list index out of range", "traceback": "Traceback...\nIndexError"}


async def test_context_limit_failure_still_trains_at_reward_zero(caplog, recorded):
    loop = _make_loop()  # max_context_tokens = response_length = 32
    first = [{"role": "user", "content": "hi"}]
    reply = {"role": "assistant", "content": "hello world"}
    overflow = first + [reply] + [{"role": "user", "content": "more"}] * 15  # 36 prompt tokens >= 32
    invoke = _wire_attempts(
        loop,
        [
            {
                "messages_per_turn": [first, overflow],
                "result": {"status_code": 500, "stop_reason": "ContextWindowOverflow"},
            }
        ],
    )
    with caplog.at_level(logging.WARNING):
        outputs = await loop.run({}, **_ROW)

    assert invoke.statuses[0][0] == 200 and invoke.statuses[0][1] != 200  # the gateway refused the 2nd turn
    assert len(invoke.calls) == 1
    assert outputs[0].reward_score == 0.0
    assert outputs[0].extra_fields["rollout_failure_class"] == "model"
    assert outputs[0].extra_fields["reward_extra_info"]["acr_failed"] == 1.0
    [line] = _failure_lines(caplog)
    assert (line["cls"], line["action"], line["step"]) == ("model", "train", "7")
    assert recorded == [(7, "model", "train")]


async def test_length_finished_last_turn_is_a_model_failure(caplog):
    from verl.workers.rollout.replica import TokenOutput

    llm = FakeLLMServerClient([TokenOutput(token_ids=[101, 102], log_probs=[-0.5, -0.6], stop_reason="length")])
    loop = _make_loop(llm)
    _wire_attempts(loop, [{"turns": 1, "result": {"status_code": 500, "stop_reason": "MaxTokensReached"}}])
    with caplog.at_level(logging.WARNING):
        outputs = await loop.run({}, **_ROW)
    assert outputs[0].reward_score == 0.0
    assert [(line["cls"], line["action"]) for line in _failure_lines(caplog)] == [("model", "train")]


async def test_agent_error_after_model_turns_drops_without_retry(caplog, recorded):
    loop = _make_loop()
    invoke = _wire_attempts(loop, [{"turns": 2, "result": _INDEX_ERROR}])
    with caplog.at_level(logging.WARNING):
        with pytest.raises(al.RolloutDropped) as excinfo:
            await loop.run({}, **_ROW)

    assert excinfo.value.failure_class == "agent_error"
    assert isinstance(excinfo.value, RuntimeError)  # verl drops just this trajectory
    assert len(invoke.calls) == 1
    [line] = _failure_lines(caplog)
    assert (line["cls"], line["action"]) == ("agent_error", "drop")
    assert line["sid"] == invoke.calls[0]["session_id"]
    assert "\n" not in line["reason"] and line["reason"].startswith("agent returned status_code=500")
    assert recorded == [(7, "agent_error", "drop")]
    assert loop._gateway.gateway.manager.turn_count(line["sid"]) == 0  # session drained


async def test_invoke_error_retries_once_on_a_fresh_session_then_succeeds(caplog, recorded):
    loop = _make_loop()
    invoke = _wire_attempts(loop, [{"invoke_error": RuntimeError("ThrottlingException")}, {"turns": 1, "result": _OK}])
    with caplog.at_level(logging.WARNING):
        outputs = await loop.run({}, **_ROW)

    first_sid, second_sid = (c["session_id"] for c in invoke.calls)
    assert first_sid != second_sid
    assert invoke.calls[1]["api_key"] == second_sid
    assert outputs[0].reward_score == 1.0
    assert outputs[0].extra_fields["acr_session_id"] == second_sid
    assert outputs[0].extra_fields["rollout_attempts"] == 2
    assert "rollout_failure_class" not in outputs[0].extra_fields
    [line] = _failure_lines(caplog)
    assert (line["cls"], line["action"], line["sid"]) == ("transient", "retry", first_sid)
    assert recorded == [(7, "transient", "retry")]
    # the first session is gone from every adapter
    assert all(first_sid not in a.store for a in loop._gateway.gateway.adapters)


async def test_status_500_without_model_turns_is_transient():
    loop = _make_loop()
    invoke = _wire_attempts(
        loop, [{"turns": 0, "result": {"status_code": 500, "stop_reason": "cold start"}}, {"turns": 1, "result": _OK}]
    )
    outputs = await loop.run({}, **_ROW)
    assert len(invoke.calls) == 2
    assert outputs[0].reward_score == 1.0


async def test_poll_error_retries_and_cancels_the_old_session():
    loop = _make_loop()
    invoke = _wire_attempts(loop, [{"turns": 1, "poll_error": RuntimeError("S3 503")}, {"turns": 1, "result": _OK}])
    outputs = await loop.run({}, **_ROW)
    assert len(invoke.calls) == 2
    invoke.futures[0].cancel_async.assert_awaited_once()
    assert outputs[0].extra_fields["rollout_attempts"] == 2
    assert sum(outputs[0].response_mask) == 2  # only the retried session's turn trains


async def test_transient_failure_drops_once_retries_are_exhausted(caplog, recorded):
    loop = _make_loop(max_rollout_retries=1)
    invoke = _wire_attempts(loop, [{"invoke_error": RuntimeError("boom 1")}, {"invoke_error": RuntimeError("boom 2")}])
    with caplog.at_level(logging.WARNING):
        with pytest.raises(al.RolloutDropped, match="boom 2") as excinfo:
            await loop.run({}, **_ROW)
    assert excinfo.value.failure_class == "transient"
    assert len(invoke.calls) == 2
    assert [(line["cls"], line["action"]) for line in _failure_lines(caplog)] == [
        ("transient", "retry"),
        ("transient", "drop"),
    ]
    assert recorded == [(7, "transient", "retry"), (7, "transient", "drop")]


async def test_zero_retries_drops_transient_immediately():
    loop = _make_loop(max_rollout_retries=0)
    invoke = _wire_attempts(loop, [{"invoke_error": RuntimeError("boom")}])
    with pytest.raises(al.RolloutDropped):
        await loop.run({}, **_ROW)
    assert len(invoke.calls) == 1


async def test_retries_share_one_max_rollout_time_deadline():
    loop = _make_loop(max_rollout_time=30.0)
    invoke = _wire_attempts(loop, [{"invoke_error": RuntimeError("boom")}, {"turns": 1, "result": _OK}])
    # start, the retry's deadline check, the second attempt's remaining-time check, ...
    clock = iter([100.0, 104.0, 112.0])
    # Replace only agent_loop's clock: the event loop and aiohttp keep the real one.
    with patch.object(al, "time", SimpleNamespace(monotonic=lambda: next(clock, 113.0))):
        await loop.run({}, **_ROW)
    # The retry polls only for what is left of the shared 30 s budget.
    assert invoke.timeouts == [pytest.approx(18.0)]


async def test_no_retry_when_the_backoff_would_overrun_the_deadline(monkeypatch, caplog):
    monkeypatch.setattr(al, "_RETRY_BACKOFF_S", (5.0, 5.0))
    loop = _make_loop(max_rollout_time=1.0)
    invoke = _wire_attempts(loop, [{"invoke_error": RuntimeError("boom")}])
    with caplog.at_level(logging.WARNING):
        with pytest.raises(al.RolloutDropped, match="no time left"):
            await loop.run({}, **_ROW)
    assert len(invoke.calls) == 1
    assert [(line["cls"], line["action"]) for line in _failure_lines(caplog)] == [("transient", "drop")]


async def test_timeout_drops_by_default_even_with_captured_turns(caplog):
    loop = _make_loop()
    invoke = _wire_attempts(loop, [{"turns": 2, "poll_timeout": True}])
    with caplog.at_level(logging.WARNING):
        with pytest.raises(al.RolloutDropped, match="timed out") as excinfo:
            await loop.run({}, **_ROW)
    assert excinfo.value.failure_class == "timeout"
    assert len(invoke.calls) == 1  # timeouts are never retried
    assert [(line["cls"], line["action"]) for line in _failure_lines(caplog)] == [("timeout", "drop")]


async def test_timeout_policy_penalize_trains_at_reward_zero(caplog):
    loop = _make_loop(timeout_policy="penalize")
    _wire_attempts(loop, [{"turns": 2, "poll_timeout": True}])
    with caplog.at_level(logging.WARNING):
        outputs = await loop.run({}, **_ROW)
    assert outputs[0].reward_score == 0.0
    assert outputs[0].extra_fields["rollout_failure_class"] == "model"
    assert [(line["cls"], line["action"]) for line in _failure_lines(caplog)] == [("model", "train")]


async def test_model_failure_without_tokens_is_dropped():
    loop = _make_loop(timeout_policy="penalize")
    _wire_attempts(loop, [{"turns": 0, "poll_timeout": True}])
    with pytest.raises(al.RolloutDropped) as excinfo:
        await loop.run({}, **_ROW)
    assert excinfo.value.failure_class == "model"


async def test_failure_reason_is_single_line_and_truncated(caplog):
    loop = _make_loop()
    _wire_attempts(loop, [{"turns": 1, "result": {"status_code": 500, "stop_reason": "line1\nline2\r\n" + "x" * 500}}])
    with caplog.at_level(logging.WARNING):
        with pytest.raises(al.RolloutDropped):
            await loop.run({}, **_ROW)
    [line] = _failure_lines(caplog)
    assert len(line["reason"]) == 200
    assert "line1 line2 " in line["reason"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"timeout_policy": "retry"}, "timeout_policy"),
        ({"max_rollout_retries": -1}, "max_rollout_retries"),
        ({"max_rollout_retries": True}, "max_rollout_retries"),
    ],
)
async def test_invalid_failure_policy_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _make_loop(**kwargs)
