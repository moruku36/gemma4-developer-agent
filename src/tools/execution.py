"""Subprocess execution with hard timeouts and bounded output.

Used by the agent to run tests and shell commands inside the target
repository. Deliberately conservative: commands run in their own process
group so a timeout kills the whole tree, output is head/tail truncated,
and nothing here assumes network access.
"""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass


@dataclass
class ExecResult:
    """Outcome of one command execution."""

    returncode: int | None  # None when the process was killed (timeout)
    stdout: str
    stderr: str
    timed_out: bool = False
    duration_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.timed_out and self.returncode == 0

    def render(self) -> str:
        """Compact text form fed back to the model."""
        status = "TIMEOUT" if self.timed_out else f"exit={self.returncode}"
        parts = [f"[{status}]"]
        if self.stdout:
            parts.append(f"stdout:\n{self.stdout}")
        if self.stderr:
            parts.append(f"stderr:\n{self.stderr}")
        return "\n".join(parts)


def truncate_middle(text: str, max_chars: int) -> str:
    """Keep head and tail of *text*, eliding the middle with a marker."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max_chars // 2
    elided = len(text) - 2 * keep
    return f"{text[:keep]}\n... [{elided} chars elided] ...\n{text[-keep:]}"


def run_command(
    command: list[str] | str,
    cwd: str | os.PathLike[str],
    timeout_seconds: float = 300.0,
    max_output_chars: int = 8000,
    env_extra: dict[str, str] | None = None,
) -> ExecResult:
    """Run *command* in *cwd*, killing the process group on timeout.

    ``command`` may be a list (preferred, no shell) or a string (run via
    ``/bin/sh -c``). Returns an :class:`ExecResult`; never raises for a
    non-zero exit code.
    """
    import time

    env = dict(os.environ)
    # Keep runs hermetic and quiet: no bytecode litter, no pip prompts.
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PIP_NO_INPUT", "1")
    if env_extra:
        env.update(env_extra)

    if isinstance(command, str):
        argv: list[str] | str = command
        use_shell = True
    else:
        argv = list(command)
        use_shell = False

    start = time.monotonic()
    proc = subprocess.Popen(
        argv,
        cwd=str(cwd),
        shell=use_shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,  # own process group -> killpg reaches children
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
        returncode: int | None = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = None
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        stdout, stderr = proc.communicate()

    return ExecResult(
        returncode=returncode,
        stdout=truncate_middle(stdout or "", max_output_chars),
        stderr=truncate_middle(stderr or "", max_output_chars),
        timed_out=timed_out,
        duration_seconds=time.monotonic() - start,
    )
