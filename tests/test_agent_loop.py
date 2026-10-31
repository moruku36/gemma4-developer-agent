"""Unit tests for the agent loop using a scripted fake LLM client.

No GPU, no vLLM: FakeClient replays canned replies implementing the same
ChatClient protocol the real vLLM client uses.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent.core import AgentConfig, AgentLoop, extract_json_object
from engine.client import ChatResponse


class FakeClient:
    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls = 0

    def chat(self, messages, *, temperature=0.2, max_tokens=4096) -> ChatResponse:
        self.calls += 1
        if not self._replies:
            return ChatResponse(text='{"thought": "done", "tool": "finish", "args": {}}')
        return ChatResponse(text=self._replies.pop(0))


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@example.com",
         "-c", "user.name=test", *args],
        check=True, capture_output=True,
    )


@pytest.fixture()
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "initial")
    return tmp_path


def _config(**overrides) -> AgentConfig:
    base = dict(max_steps=10, task_timeout_seconds=120, max_tokens_per_task=100_000)
    base.update(overrides)
    return AgentConfig(**base)


def test_successful_fix_produces_valid_patch(repo):
    client = FakeClient([
        '{"thought": "read the file", "tool": "read_file", "args": {"path": "calc.py"}}',
        '{"thought": "fix the operator", "tool": "edit_file", "args": '
        '{"path": "calc.py", "old_text": "    return a - b", "new_text": "    return a + b"}}',
        '{"thought": "done", "tool": "finish", "args": {}}',
    ])
    result = AgentLoop(client, _config(), repo).run_task("add() subtracts instead of adding")
    assert result.status == "success"
    assert "+    return a + b" in result.patch
    assert result.steps == 3


def test_step_limit_stops_runaway_agent(repo):
    client = FakeClient(
        ['{"thought": "loop", "tool": "list_dir", "args": {}}'] * 50
    )
    result = AgentLoop(client, _config(max_steps=5), repo).run_task("never finishes")
    assert result.status == "no_patch"
    assert result.steps == 5
    assert client.calls == 5


def test_token_budget_stops_agent(repo):
    class HungryClient(FakeClient):
        def chat(self, messages, *, temperature=0.2, max_tokens=4096):
            response = super().chat(messages, temperature=temperature, max_tokens=max_tokens)
            return ChatResponse(text=response.text, prompt_tokens=60_000,
                                completion_tokens=60_000)

    client = HungryClient(
        ['{"thought": "loop", "tool": "list_dir", "args": {}}'] * 10
    )
    result = AgentLoop(client, _config(max_tokens_per_task=100_000), repo).run_task("x")
    assert result.status == "no_patch"  # stopped by budget, no edits made
    assert result.steps <= 2


def test_malformed_reply_is_corrected_then_recovers(repo):
    client = FakeClient([
        "Sure! Let me look at the code first.",  # not JSON -> parse error
        '{"thought": "ok", "tool": "finish", "args": {}}',
    ])
    result = AgentLoop(client, _config(), repo).run_task("anything")
    assert result.status == "no_patch"  # finished cleanly, just no changes
    assert result.steps == 2


def test_persistent_garbage_aborts_with_error(repo):
    client = FakeClient(["not json"] * 10)
    result = AgentLoop(client, _config(max_parse_errors=2), repo).run_task("x")
    assert result.status == "error"


def test_unknown_tool_reports_error_and_continues(repo):
    client = FakeClient([
        '{"thought": "try a fake tool", "tool": "teleport", "args": {}}',
        '{"thought": "give up", "tool": "finish", "args": {}}',
    ])
    result = AgentLoop(client, _config(), repo).run_task("x")
    assert result.status == "no_patch"
    observations = [m["content"] for m in result.transcript if m["role"] == "user"]
    assert any("unknown tool" in obs for obs in observations)


def test_extract_json_object_variants():
    assert extract_json_object('{"tool": "finish", "args": {}}') == {
        "tool": "finish", "args": {}}
    fenced = 'Here you go:\n```json\n{"tool": "read_file", "args": {"path": "a.py"}}\n```'
    assert extract_json_object(fenced)["tool"] == "read_file"
    prose = 'I will call {"tool": "list_dir", "args": {}} now'
    assert extract_json_object(prose)["tool"] == "list_dir"
    assert extract_json_object("no json here") is None
