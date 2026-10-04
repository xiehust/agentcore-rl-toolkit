"""Tests for the opt-in ``require_registered_sessions`` auth guard.

Drives the adapters over the aiohttp test client (same fixture style as
test_adapters.py) to assert:

  (a) default mode: an unknown sid still implicitly opens a session (unchanged);
  (b) flag on: an unregistered Bearer is refused 401 and leaves no session state;
  (c) flag on: after create_session(sid) the same request succeeds;
  (d) flag on: unauthenticated endpoints (/healthz, /v1/models) are unaffected.
"""

import pytest
from aiohttp.test_utils import TestClient, TestServer

from agentcore_rl_toolkit.rollout_gateway.adapters import AnthropicAdapter, OpenAIAdapter
from agentcore_rl_toolkit.rollout_gateway.render import ParsedOutput
from agentcore_rl_toolkit.rollout_gateway.trajectory import TurnRecord


class FakeRenderer:
    """Deterministic word-level 'tokenizer': one id per whitespace token."""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _id(self, tok: str) -> int:
        return self.vocab.setdefault(tok, len(self.vocab) + 1)

    def _encode(self, text: str) -> list[int]:
        return [self._id(t) for t in text.split()]

    async def render(self, messages, *, tools=None, add_generation_prompt=True):
        ids: list[int] = []
        for m in messages:
            ids += self._encode(f"{m['role']}:")
            ids += self._encode(m.get("content") or "")
        if add_generation_prompt:
            ids += self._encode("assistant:")
        return ids

    def get_stop_sequences(self):
        return []

    def parse(self, output_ids, *, tools_schema=None):
        inv = {v: k for k, v in self.vocab.items()}
        text = " ".join(inv.get(i, "?") for i in output_ids)
        return ParsedOutput(reasoning="", text=text, tool_uses=[], ill_formed=False)


class FakeBackend:
    """Returns a scripted response per turn; echoes it as new vocab ids."""

    def __init__(self, renderer: FakeRenderer, replies: list[str]):
        self.renderer = renderer
        self.replies = list(replies)
        self.calls: list[list[int]] = []

    async def generate(self, *, prompt_ids, sampling_params, session_id=None, image_data=None, video_data=None):
        self.calls.append(list(prompt_ids))
        reply = self.replies.pop(0)
        out_ids = self.renderer._encode(reply)
        return TurnRecord(
            prompt_ids=list(prompt_ids),
            output_ids=out_ids,
            finish_reason="stop",
            output_log_probs=[-0.5] * len(out_ids),
        )


def _openai_body(content: str = "two plus two") -> dict:
    return {"model": "x", "messages": [{"role": "user", "content": content}]}


async def _serve(adapter):
    server = TestServer(adapter.app)
    client = TestClient(server)
    await client.start_server()
    return client


@pytest.mark.asyncio
async def test_default_mode_unknown_sid_opens_session():
    """(a) With the flag off (default), an unknown Bearer still creates a session."""
    renderer = FakeRenderer()
    backend = FakeBackend(renderer, replies=["four"])
    adapter = OpenAIAdapter(backend=backend, renderer=renderer, tokenizer=None)
    assert adapter.require_registered_sessions is False

    client = await _serve(adapter)
    try:
        sid = "stranger"
        resp = await client.post(
            "/v1/chat/completions", json=_openai_body(), headers={"Authorization": f"Bearer {sid}"}
        )
        assert resp.status == 200
        assert sid in adapter.store  # implicitly opened
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_flag_on_unregistered_bearer_rejected():
    """(b) With the flag on, an unregistered Bearer -> 401 and no session state."""
    renderer = FakeRenderer()
    backend = FakeBackend(renderer, replies=["four"])
    adapter = OpenAIAdapter(backend=backend, renderer=renderer, tokenizer=None, require_registered_sessions=True)

    client = await _serve(adapter)
    try:
        sid = "stranger"
        resp = await client.post(
            "/v1/chat/completions", json=_openai_body(), headers={"Authorization": f"Bearer {sid}"}
        )
        assert resp.status == 401
        assert (await resp.text()) == "unknown session"
        # no session or trajectory state leaked, and the backend was never called
        assert sid not in adapter.store
        assert adapter.manager.turn_count(sid) == 0
        assert backend.calls == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_flag_on_registered_sid_succeeds():
    """(c) With the flag on, a sid opened via open_session drives a turn normally."""
    renderer = FakeRenderer()
    backend = FakeBackend(renderer, replies=["four"])
    adapter = OpenAIAdapter(backend=backend, renderer=renderer, tokenizer=None, require_registered_sessions=True)

    client = await _serve(adapter)
    try:
        sid = "ep1:solver"
        adapter.open_session(sid)  # <- what RolloutGateway.create_session calls
        resp = await client.post(
            "/v1/chat/completions", json=_openai_body(), headers={"Authorization": f"Bearer {sid}"}
        )
        assert resp.status == 200
        assert (await resp.json())["choices"][0]["message"]["content"] == "four"
        assert adapter.manager.turn_count(sid) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_flag_on_unauthenticated_endpoints_unaffected():
    """(d) With the flag on, health / models probes stay open (no Bearer needed)."""
    renderer = FakeRenderer()
    backend = FakeBackend(renderer, replies=[])
    adapter = OpenAIAdapter(backend=backend, renderer=renderer, tokenizer=None, require_registered_sessions=True)

    client = await _serve(adapter)
    try:
        for path in ("/healthz", "/v1/models"):
            resp = await client.get(path)
            assert resp.status == 200
            assert (await resp.json()) == {"ok": True}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_flag_on_anthropic_adapter_rejects_unregistered():
    """The guard is in the shared handler, so the Anthropic route enforces it too."""
    renderer = FakeRenderer()
    backend = FakeBackend(renderer, replies=["four"])
    adapter = AnthropicAdapter(backend=backend, renderer=renderer, tokenizer=None, require_registered_sessions=True)

    client = await _serve(adapter)
    try:
        body = {"model": "x", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
        # unregistered -> 401
        resp = await client.post("/v1/messages", json=body, headers={"Authorization": "Bearer stranger"})
        assert resp.status == 401
        assert backend.calls == []
        # count_tokens stays unauthenticated
        ct = await client.post("/v1/messages/count_tokens", json=body)
        assert ct.status == 200
        # registered -> succeeds
        adapter.open_session("known")
        ok = await client.post("/v1/messages", json=body, headers={"Authorization": "Bearer known"})
        assert ok.status == 200
    finally:
        await client.close()
