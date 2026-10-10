"""OfficeBench ``rl_app``: an empty final message must not stop the reward from being computed.

The example's own dependencies (strands, strands_tools, its local modules) are not part of
the toolkit's environment, so they are stubbed in ``sys.modules`` before importing the app;
``rollout_entrypoint`` is replaced by identity so the handler can be called directly.
"""

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentcore_rl_toolkit import AgentCoreRLApp

APP_DIR = Path(__file__).resolve().parents[2] / "examples" / "strands_officebench_agent"
_STUBBED = [
    "dotenv",
    "models",
    "reward",
    "strands",
    "strands.agent",
    "strands.agent.conversation_manager",
    "strands.models",
    "strands.models.openai",
    "strands_tools",
    "tools",
    "utils",
]


@pytest.fixture
def rl_app(monkeypatch):
    stubs = {name: types.ModuleType(name) for name in _STUBBED}
    stubs["dotenv"].load_dotenv = lambda *a, **k: None
    stubs["models"].InvocationRequest = lambda task_uri, testbed_uri=None, **_: SimpleNamespace(
        task_uri=task_uri, testbed_uri=testbed_uri
    )
    stubs["reward"].OfficeBenchReward = lambda: None
    stubs["strands"].Agent = None
    stubs["strands.agent.conversation_manager"].NullConversationManager = lambda: None
    stubs["strands.models"].BedrockModel = lambda **kwargs: None
    stubs["strands.models.openai"].OpenAIModel = lambda **kwargs: None
    stubs["strands_tools"].shell = None
    stubs["tools"].ALL_TOOLS = []
    stubs["utils"].load_task_from_s3 = None
    stubs["utils"].setup_testbed = None
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "rl_app", raising=False)
    monkeypatch.syspath_prepend(str(APP_DIR))
    monkeypatch.setattr(AgentCoreRLApp, "rollout_entrypoint", lambda self, func: func)
    module = importlib.import_module("rl_app")
    yield module
    sys.modules.pop("rl_app", None)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ({"role": "assistant", "content": []}, ""),
        ({"role": "assistant"}, ""),
        (None, ""),
        ({"content": [{"toolUse": {"name": "shell"}}]}, ""),
        ({"content": [{"text": "done"}, {"text": "!"}]}, "done!"),
    ],
)
def test_response_text_never_indexes_into_content(rl_app, message, expected):
    assert rl_app._response_text(message) == expected


def test_empty_final_message_still_computes_the_reward(rl_app, monkeypatch):
    class FakeAgent:
        def __init__(self, **kwargs):
            self.messages = [{"role": "user", "content": [{"text": "task"}]}, {"role": "assistant", "content": []}]

        def __call__(self, user_input):
            return SimpleNamespace(message={"role": "assistant", "content": []})

    reward_calls = []

    def reward_fn(**kwargs):
        reward_calls.append(kwargs)
        return 1.0

    monkeypatch.setattr(rl_app, "Agent", FakeAgent)
    monkeypatch.setattr(rl_app, "reward_fn", reward_fn)
    monkeypatch.setattr(rl_app, "load_task_from_s3", lambda uri: {"task": "do it", "evaluation": [{"function": "f"}]})
    monkeypatch.setattr(rl_app, "setup_testbed", lambda uri: None)

    result = rl_app.invoke_agent(
        {"task_uri": "s3://bucket/task.json", "_rollout": {"base_url": "http://gw/v1", "model_id": "m"}}
    )

    assert result["rewards"] == 1.0
    assert reward_calls == [{"testbed_dir": "/testbed", "evaluation_config": [{"function": "f"}]}]
