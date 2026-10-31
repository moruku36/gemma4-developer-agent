#!/usr/bin/env python3
"""Local evaluation entry point.

Modes:
  --tasks tasks.json   Run the agent (backed by a local vLLM server) over
                       real task repos and write results + patches.
  --demo               Harness smoke test: builds two tiny buggy repos and
                       drives the loop with a scripted client, proving the
                       explore->edit->test->patch pipeline end to end
                       without any GPU or model.

tasks.json format: [{"task_id": ..., "repo_path": ..., "issue_text": ...,
                     "test_cmd": "python3 -m pytest -q"  # optional
                    }]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agent.core import AgentConfig  # noqa: E402
from engine.client import ChatResponse, LocalVLLMClient  # noqa: E402
from pipeline.runner import TaskSpec, run_tasks  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "default_agent.yaml"


# ---------------------------------------------------------------- demo mode
class ScriptedClient:
    """ChatClient that replays canned replies (harness smoke test only)."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)

    def chat(self, messages, *, temperature=0.2, max_tokens=4096) -> ChatResponse:
        if not self._replies:
            return ChatResponse(text='{"thought": "out of script", "tool": "finish", "args": {}}')
        return ChatResponse(text=self._replies.pop(0))


def _git_init_repo(path: Path, files: dict[str, str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(path), "-c", "user.email=demo@example.com",
             "-c", "user.name=demo", *args],
            check=True, capture_output=True,
        )
    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "initial")


_DEMO_TASKS: list[dict] = [
    {
        "task_id": "demo-calc",
        "files": {
            "conftest.py": "",
            "calc.py": "def add(a, b):\n    return a - b\n",
            "tests/test_calc.py": (
                "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
            ),
        },
        "issue_text": "add(a, b) returns the difference instead of the sum: "
                      "add(2, 3) gives -1 but must give 5.",
        "test_cmd": "python3 -m pytest -q",
        "replies": [
            '{"thought": "Plan: read calc.py, fix add, run tests.", '
            '"tool": "read_file", "args": {"path": "calc.py"}}',
            '{"thought": "add() subtracts; it must add.", "tool": "edit_file", '
            '"args": {"path": "calc.py", "old_text": "    return a - b", '
            '"new_text": "    return a + b"}}',
            '{"thought": "Verify.", "tool": "run_command", '
            '"args": {"command": "python3 -m pytest -q"}}',
            '{"thought": "Tests pass; fix complete.", "tool": "finish", "args": {}}',
        ],
    },
    {
        "task_id": "demo-text",
        "files": {
            "conftest.py": "",
            "textutil.py": (
                'def slugify(text):\n    return text.replace(" ", "-")\n'
            ),
            "tests/test_textutil.py": (
                "from textutil import slugify\n\n\n"
                "def test_slugify():\n    assert slugify('Hello World') == 'hello-world'\n"
            ),
        },
        "issue_text": "slugify('Hello World') returns 'Hello-World' but slugs "
                      "must be lowercase: 'hello-world'.",
        "test_cmd": "python3 -m pytest -q",
        "replies": [
            '{"thought": "Plan: read textutil.py, lowercase in slugify, test.", '
            '"tool": "read_file", "args": {"path": "textutil.py"}}',
            '{"thought": "slugify never lowercases.", "tool": "edit_file", '
            '"args": {"path": "textutil.py", '
            '"old_text": "    return text.replace(\\" \\", \\"-\\")", '
            '"new_text": "    return text.lower().replace(\\" \\", \\"-\\")"}}',
            '{"thought": "Verify.", "tool": "run_command", '
            '"args": {"command": "python3 -m pytest -q"}}',
            '{"thought": "Tests pass; fix complete.", "tool": "finish", "args": {}}',
        ],
    },
]


def run_demo(config_path: Path, output_dir: Path) -> int:
    staging = Path(tempfile.mkdtemp(prefix="gemma4-demo-"))
    tasks: list[TaskSpec] = []
    scripts: dict[str, list[str]] = {}
    for spec in _DEMO_TASKS:
        repo = staging / spec["task_id"]
        _git_init_repo(repo, spec["files"])
        tasks.append(TaskSpec(
            task_id=spec["task_id"], repo_path=str(repo),
            issue_text=spec["issue_text"], test_cmd=spec["test_cmd"],
        ))
        scripts[spec["task_id"]] = spec["replies"]
    summary = run_tasks(
        tasks,
        client_factory=lambda task: ScriptedClient(scripts[task.task_id]),
        agent_config=AgentConfig.from_yaml(config_path),
        config_path=config_path,
        output_dir=output_dir,
    )
    print(json.dumps(summary.counts, indent=2))
    for outcome in summary.outcomes:
        print(f"{outcome.task_id}: {outcome.status} "
              f"(local tests: {outcome.local_test_status or 'n/a'})")
    return 0 if all(o.status == "success" for o in summary.outcomes) else 1


# ---------------------------------------------------------------- real mode
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, help="tasks.json path")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts" / "local-eval")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="gemma-4")
    parser.add_argument("--demo", action="store_true", help="run the harness smoke test")
    args = parser.parse_args(argv)

    if args.demo:
        return run_demo(args.config, args.output)
    if args.tasks is None:
        parser.error("--tasks is required unless --demo is given")

    tasks = TaskSpec.load_many(args.tasks)
    client = LocalVLLMClient(base_url=args.base_url, model=args.model)
    summary = run_tasks(
        tasks,
        client_factory=lambda task: client,
        agent_config=AgentConfig.from_yaml(args.config),
        config_path=args.config,
        output_dir=args.output,
    )
    print(json.dumps(summary.counts, indent=2))
    print(f"results: {args.output / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
