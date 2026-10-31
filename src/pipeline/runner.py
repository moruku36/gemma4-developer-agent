"""Batch evaluation runner: many tasks, one global wall-clock budget.

Each task runs in a *disposable copy* of its repository, so the source
checkout stays pristine and tasks cannot contaminate each other. Results
(status, patch, token/step counts, timing) are written as JSON plus one
``.patch`` file per task — the same artifacts the Kaggle notebook emits
as its predictions file.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from agent.core import AgentConfig, AgentLoop
from engine.client import ChatClient
from tools.execution import run_command

ClientFactory = Callable[["TaskSpec"], ChatClient]


@dataclass
class TaskSpec:
    task_id: str
    repo_path: str  # pristine checkout of the target repository
    issue_text: str
    # Optional local-eval check: command run in the workspace after a
    # successful patch (e.g. the task's public tests). Never used for
    # hidden-test scoring; diagnostics only.
    test_cmd: str | None = None

    @classmethod
    def load_many(cls, path: str | Path) -> list["TaskSpec"]:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        return [cls(**item) for item in raw]


@dataclass
class TaskOutcome:
    task_id: str
    status: str
    steps: int = 0
    tokens_used: int = 0
    duration_seconds: float = 0.0
    patch_file: str = ""
    message: str = ""
    validation_problems: list[str] = field(default_factory=list)
    local_test_status: str = ""  # "", "passed", "failed", "timeout"


@dataclass
class RunSummary:
    outcomes: list[TaskOutcome]
    output_dir: Path

    @property
    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for outcome in self.outcomes:
            result[outcome.status] = result.get(outcome.status, 0) + 1
        return result


def _load_run_limits(config_path: str | Path) -> tuple[float, float]:
    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    run = raw.get("run", {})
    return (
        float(run.get("global_timeout_seconds", 41400)),
        float(run.get("reserve_seconds", 900)),
    )


def prepare_workspace(task: TaskSpec, workspaces_root: Path) -> Path:
    """Copy the task repository into a disposable workspace."""
    workspace = workspaces_root / task.task_id
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(
        task.repo_path,
        workspace,
        ignore=shutil.ignore_patterns("__pycache__", ".venv", "node_modules"),
    )
    return workspace


def run_tasks(
    tasks: list[TaskSpec],
    client_factory: ClientFactory,
    agent_config: AgentConfig,
    config_path: str | Path,
    output_dir: str | Path,
) -> RunSummary:
    """Run tasks sequentially until the global budget is nearly exhausted."""
    output_dir = Path(output_dir)
    patches_dir = output_dir / "patches"
    workspaces_root = output_dir / "workspaces"
    patches_dir.mkdir(parents=True, exist_ok=True)
    workspaces_root.mkdir(parents=True, exist_ok=True)

    global_timeout, reserve = _load_run_limits(config_path)
    deadline = time.monotonic() + global_timeout
    outcomes: list[TaskOutcome] = []

    for task in tasks:
        remaining = deadline - time.monotonic()
        if remaining <= reserve:
            outcomes.append(TaskOutcome(
                task_id=task.task_id, status="skipped_budget",
                message=f"global budget exhausted ({remaining:.0f}s left)",
            ))
            continue
        started = time.monotonic()
        workspace = prepare_workspace(task, workspaces_root)
        loop = AgentLoop(client_factory(task), agent_config, workspace)
        result = loop.run_task(task.issue_text, repo_hint=task.task_id)
        outcome = TaskOutcome(
            task_id=task.task_id,
            status=result.status,
            steps=result.steps,
            tokens_used=result.tokens_used,
            duration_seconds=time.monotonic() - started,
            message=result.message,
            validation_problems=result.validation_problems,
        )
        if result.patch:
            patch_path = patches_dir / f"{task.task_id}.patch"
            patch_path.write_text(result.patch, encoding="utf-8")
            outcome.patch_file = str(patch_path)
            if task.test_cmd:
                check = run_command(
                    task.test_cmd, cwd=workspace,
                    timeout_seconds=agent_config.command_timeout_seconds,
                    max_output_chars=2000,
                )
                outcome.local_test_status = (
                    "timeout" if check.timed_out
                    else "passed" if check.returncode == 0 else "failed"
                )
        outcomes.append(outcome)

    summary = RunSummary(outcomes=outcomes, output_dir=output_dir)
    (output_dir / "results.json").write_text(
        json.dumps(
            {"counts": summary.counts, "outcomes": [asdict(o) for o in outcomes]},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return summary
