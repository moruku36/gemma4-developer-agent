"""The agent loop: ReAct-style turns over a strict JSON tool protocol.

Determinism controls live here, not in the model:

- ``max_steps``         — hard cap on model turns per task
- ``max_tokens_per_task`` — cumulative token budget (server-reported usage
  when available, character estimate otherwise)
- ``task_timeout_seconds`` — wall-clock deadline per task
- ``max_parse_errors``  — malformed replies tolerated before aborting

Whatever happens inside the loop, a task ends by generating the working
tree diff and validating it (applies cleanly + touched Python compiles);
only then is the patch returned as a success.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from engine.client import ChatClient
from tools import patch as patch_tools
from tools.execution import run_command
from tools.file_ops import FileOps, FileOpsError
from . import prompts
from .memory import ConversationMemory, estimate_tokens

ToolFn = Callable[[dict[str, Any]], str]


@dataclass
class AgentConfig:
    max_steps: int = 60
    task_timeout_seconds: float = 1800.0
    max_tokens_per_task: int = 400_000
    temperature: float = 0.2
    max_completion_tokens: int = 4096
    max_parse_errors: int = 3
    max_context_tokens: int = 28_000
    tool_output_max_chars: int = 6000
    keep_recent_turns: int = 6
    command_timeout_seconds: float = 300.0
    command_output_max_chars: int = 8000
    read_max_lines: int = 400

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentConfig":
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        agent = raw.get("agent", {})
        context = raw.get("context", {})
        tools = raw.get("tools", {})
        return cls(
            max_steps=int(agent.get("max_steps", 60)),
            task_timeout_seconds=float(agent.get("task_timeout_seconds", 1800)),
            max_tokens_per_task=int(agent.get("max_tokens_per_task", 400_000)),
            temperature=float(agent.get("temperature", 0.2)),
            max_completion_tokens=int(agent.get("max_completion_tokens", 4096)),
            max_parse_errors=int(agent.get("max_parse_errors", 3)),
            max_context_tokens=int(context.get("max_context_tokens", 28_000)),
            tool_output_max_chars=int(context.get("tool_output_max_chars", 6000)),
            keep_recent_turns=int(context.get("keep_recent_turns", 6)),
            command_timeout_seconds=float(tools.get("command_timeout_seconds", 300)),
            command_output_max_chars=int(tools.get("command_output_max_chars", 8000)),
            read_max_lines=int(tools.get("read_max_lines", 400)),
        )


@dataclass
class TaskResult:
    status: str  # success | no_patch | invalid_patch | timeout | step_limit | token_budget | error
    patch: str = ""
    steps: int = 0
    tokens_used: int = 0
    message: str = ""
    validation_problems: list[str] = field(default_factory=list)
    transcript: list[dict[str, str]] = field(default_factory=list)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first balanced JSON object from a model reply.

    Tolerates markdown fences and leading prose; returns None when no
    parseable object exists.
    """
    text = text.strip()
    candidates: list[str] = []
    if "```" in text:
        for chunk in text.split("```"):
            chunk = chunk.strip()
            if chunk.startswith("json"):
                chunk = chunk[4:].strip()
            if chunk.startswith("{"):
                candidates.append(chunk)
    candidates.append(text)
    start = text.find("{")
    if start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


class ToolRegistry:
    """Name -> callable dispatch with uniform error reporting."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        self._tools[name] = fn

    def names(self) -> list[str]:
        return sorted(self._tools)

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        fn = self._tools.get(name)
        if fn is None:
            return f"ERROR: unknown tool {name!r}; available: {', '.join(self.names())}"
        try:
            return fn(args)
        except FileOpsError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:  # tool bugs must not kill the whole task
            return f"ERROR: {type(exc).__name__}: {exc}"


class AgentLoop:
    """Runs one issue-fixing task against one repository checkout."""

    def __init__(self, client: ChatClient, config: AgentConfig, repo_root: str | Path):
        self.client = client
        self.config = config
        self.repo_root = Path(repo_root)
        self.file_ops = FileOps(self.repo_root, read_max_lines=config.read_max_lines)
        self.registry = ToolRegistry()
        self._register_tools()

    # -- tool wiring -----------------------------------------------------
    def _register_tools(self) -> None:
        ops, cfg = self.file_ops, self.config
        self.registry.register(
            "list_dir",
            lambda a: ops.list_dir(a.get("path", "."), bool(a.get("recursive", False))),
        )
        self.registry.register(
            "read_file",
            lambda a: ops.read_file(
                a["path"], int(a.get("start_line", 1)), a.get("end_line")
            ),
        )
        self.registry.register(
            "search", lambda a: ops.search(a["pattern"], a.get("path", "."))
        )
        self.registry.register("ast_outline", lambda a: ops.ast_outline(a["path"]))
        self.registry.register(
            "edit_file",
            lambda a: ops.edit_file(a["path"], a["old_text"], a["new_text"]),
        )
        self.registry.register(
            "write_file", lambda a: ops.write_file(a["path"], a["content"])
        )

        def _run_command(args: dict[str, Any]) -> str:
            timeout = float(args.get("timeout_seconds", cfg.command_timeout_seconds))
            result = run_command(
                str(args["command"]),
                cwd=self.repo_root,
                timeout_seconds=min(timeout, cfg.command_timeout_seconds),
                max_output_chars=cfg.command_output_max_chars,
            )
            return result.render()

        self.registry.register("run_command", _run_command)

    # -- main loop -------------------------------------------------------
    def run_task(self, issue_text: str, repo_hint: str = "") -> TaskResult:
        cfg = self.config
        memory = ConversationMemory(
            max_context_tokens=cfg.max_context_tokens,
            tool_output_max_chars=cfg.tool_output_max_chars,
            keep_recent_turns=cfg.keep_recent_turns,
        )
        memory.add("system", prompts.build_system_prompt())
        memory.add("user", prompts.build_task_prompt(issue_text, repo_hint))

        deadline = time.monotonic() + cfg.task_timeout_seconds
        steps = 0
        tokens_used = 0
        parse_errors = 0
        stop_reason = "step_limit"

        while steps < cfg.max_steps:
            if time.monotonic() > deadline:
                stop_reason = "timeout"
                break
            if tokens_used > cfg.max_tokens_per_task:
                stop_reason = "token_budget"
                break
            memory.prune_if_needed()
            try:
                response = self.client.chat(
                    memory.messages,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_completion_tokens,
                )
            except Exception as exc:
                return TaskResult(
                    status="error",
                    steps=steps,
                    tokens_used=tokens_used,
                    message=f"LLM call failed: {type(exc).__name__}: {exc}",
                    transcript=memory.messages,
                )
            steps += 1
            tokens_used += response.total_tokens or estimate_tokens(response.text)
            memory.add("assistant", response.text)

            action = extract_json_object(response.text)
            if action is None or "tool" not in action:
                parse_errors += 1
                if parse_errors > cfg.max_parse_errors:
                    stop_reason = "error"
                    break
                memory.add_observation(
                    "Your reply was not a valid tool-call JSON object. Reply with "
                    "exactly one JSON object in the required format."
                )
                continue

            tool = str(action.get("tool"))
            args = action.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            if tool == "finish":
                stop_reason = "finished"
                break
            memory.add_observation(self.registry.dispatch(tool, args))
        else:
            stop_reason = "step_limit"

        return self._finalize(stop_reason, steps, tokens_used, memory)

    # -- patch extraction --------------------------------------------------
    def _finalize(
        self,
        stop_reason: str,
        steps: int,
        tokens_used: int,
        memory: ConversationMemory,
    ) -> TaskResult:
        try:
            diff = patch_tools.generate_diff(self.repo_root)
        except patch_tools.PatchError as exc:
            return TaskResult(
                status="error", steps=steps, tokens_used=tokens_used,
                message=str(exc), transcript=memory.messages,
            )
        if not diff.strip():
            status = stop_reason if stop_reason in ("timeout", "error") else "no_patch"
            return TaskResult(
                status=status, steps=steps, tokens_used=tokens_used,
                message=f"agent stopped ({stop_reason}) without changing any file",
                transcript=memory.messages,
            )
        ok, problems = patch_tools.validate_patch(self.repo_root, diff)
        return TaskResult(
            status="success" if ok else "invalid_patch",
            patch=diff if ok else "",
            steps=steps,
            tokens_used=tokens_used,
            message=f"agent stopped ({stop_reason}); patch validation "
                    f"{'passed' if ok else 'failed'}",
            validation_problems=problems,
            transcript=memory.messages,
        )
