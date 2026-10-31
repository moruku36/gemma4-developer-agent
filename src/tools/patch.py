"""Patch generation and validation against a git working tree.

The agent edits files in place; at the end of a task the harness turns the
working-tree changes into a unified diff and validates it twice:

1. ``git apply --check`` against a pristine clone at the original HEAD, and
2. a Python ``compile()`` syntax check of every touched ``.py`` file.

Only patches passing both checks are handed to the scorer.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from .execution import run_command


class PatchError(Exception):
    """Raised when diff generation or validation fails hard."""


def _git(repo: Path, *args: str, timeout: float = 60.0):
    return run_command(["git", "-C", str(repo), *args], cwd=repo, timeout_seconds=timeout)


def changed_files(repo: str | Path) -> list[str]:
    """Paths (relative) of modified/addeded/deleted files vs HEAD."""
    repo = Path(repo)
    result = _git(repo, "status", "--porcelain")
    if result.returncode != 0:
        raise PatchError(f"git status failed: {result.stderr.strip()}")
    files: list[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if " -> " in path:  # rename: keep the destination
            path = path.split(" -> ", 1)[1]
        files.append(path.strip('"'))
    return files


def generate_diff(repo: str | Path) -> str:
    """Unified diff of all working-tree changes (incl. untracked files)."""
    repo = Path(repo)
    chunks: list[str] = []
    tracked = _git(repo, "diff", "HEAD", "--no-color")
    if tracked.returncode != 0:
        raise PatchError(f"git diff failed: {tracked.stderr.strip()}")
    if tracked.stdout.strip():
        chunks.append(tracked.stdout.rstrip("\n"))
    # Untracked files never appear in `git diff`; diff each against /dev/null.
    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    for line in status.stdout.splitlines():
        if line.startswith("?? "):
            rel = line[3:].strip().strip('"')
            new_file = _git(repo, "diff", "--no-color", "--no-index", "--",
                            "/dev/null", rel)
            if new_file.stdout.strip():
                chunks.append(new_file.stdout.rstrip("\n"))
    return ("\n".join(chunks) + "\n") if chunks else ""


def syntax_check(repo: str | Path, files: list[str]) -> list[str]:
    """Return a list of syntax-error strings for touched Python files."""
    repo = Path(repo)
    errors: list[str] = []
    for rel in files:
        if not rel.endswith(".py"):
            continue
        path = repo / rel
        if not path.is_file():  # deleted file: nothing to compile
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        try:
            compile(source, rel, "exec")
        except SyntaxError as exc:
            errors.append(f"{rel}: {exc}")
    return errors


def check_patch_applies(repo: str | Path, patch_text: str) -> tuple[bool, str]:
    """Verify *patch_text* applies cleanly on a pristine clone at HEAD."""
    repo = Path(repo)
    if not patch_text.strip():
        return False, "patch is empty"
    tmp = Path(tempfile.mkdtemp(prefix="patchcheck-"))
    try:
        clone = run_command(
            ["git", "clone", "--quiet", "--no-hardlinks", str(repo), str(tmp / "clone")],
            cwd=tmp, timeout_seconds=120,
        )
        if clone.returncode != 0:
            return False, f"clone failed: {clone.stderr.strip()}"
        patch_file = tmp / "candidate.patch"
        patch_file.write_text(patch_text, encoding="utf-8")
        check = run_command(
            ["git", "-C", str(tmp / "clone"), "apply", "--check", str(patch_file)],
            cwd=tmp, timeout_seconds=60,
        )
        if check.returncode == 0:
            return True, "applies cleanly"
        return False, f"git apply --check failed: {check.stderr.strip()}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def validate_patch(repo: str | Path, patch_text: str) -> tuple[bool, list[str]]:
    """Full validation: non-empty, applies cleanly, touched .py compiles."""
    problems: list[str] = []
    if not patch_text.strip():
        return False, ["patch is empty"]
    ok, message = check_patch_applies(repo, patch_text)
    if not ok:
        problems.append(message)
    problems.extend(syntax_check(repo, changed_files(repo)))
    return (not problems), problems
