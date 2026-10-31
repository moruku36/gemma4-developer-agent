"""File exploration and editing tools, confined to a repository root.

Every path the model supplies is resolved and checked to stay inside the
root, so a buggy (or adversarial) issue text cannot make the agent read or
overwrite files elsewhere on the notebook filesystem.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

from .execution import run_command, truncate_middle

_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".tox"}


class FileOpsError(Exception):
    """Raised for any tool-level failure; the agent loop reports it back."""


class FileOps:
    """File tools bound to one repository root."""

    def __init__(self, root: str | os.PathLike[str], read_max_lines: int = 400):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileOpsError(f"repository root is not a directory: {root}")
        self.read_max_lines = read_max_lines

    # -- path safety -----------------------------------------------------
    def _resolve(self, rel_path: str) -> Path:
        path = (self.root / rel_path).resolve()
        if path != self.root and self.root not in path.parents:
            raise FileOpsError(f"path escapes repository root: {rel_path}")
        return path

    # -- reading ---------------------------------------------------------
    def list_dir(self, rel_path: str = ".", recursive: bool = False) -> str:
        base = self._resolve(rel_path)
        if not base.is_dir():
            raise FileOpsError(f"not a directory: {rel_path}")
        entries: list[str] = []
        if recursive:
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
                for name in sorted(filenames):
                    entries.append(str((Path(dirpath) / name).relative_to(self.root)))
        else:
            for child in sorted(base.iterdir(), key=lambda p: p.name):
                if child.name in _SKIP_DIRS:
                    continue
                suffix = "/" if child.is_dir() else ""
                entries.append(str(child.relative_to(self.root)) + suffix)
        return "\n".join(entries) if entries else "(empty)"

    def read_file(self, rel_path: str, start_line: int = 1, end_line: int | None = None) -> str:
        path = self._resolve(rel_path)
        if not path.is_file():
            raise FileOpsError(f"not a file: {rel_path}")
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            raise FileOpsError(f"cannot read {rel_path}: {exc}") from exc
        start = max(start_line, 1)
        end = min(end_line if end_line is not None else start + self.read_max_lines - 1,
                  start + self.read_max_lines - 1, len(lines))
        if start > len(lines):
            return f"(file has {len(lines)} lines; start_line {start_line} is past EOF)"
        numbered = [f"{i:>6}\t{lines[i - 1]}" for i in range(start, end + 1)]
        header = f"# {rel_path} (lines {start}-{end} of {len(lines)})"
        return header + "\n" + "\n".join(numbered)

    # -- searching -------------------------------------------------------
    def search(self, pattern: str, rel_path: str = ".", max_results: int = 60) -> str:
        """Search for a regex; uses ripgrep when available, else pure Python."""
        base = self._resolve(rel_path)
        rg_result = None
        try:
            rg_result = run_command(
                ["rg", "--no-heading", "--line-number", "--max-count", "20",
                 "-e", pattern, "."],
                cwd=base, timeout_seconds=30,
            )
        except FileNotFoundError:
            rg_result = None  # no ripgrep binary -> Python fallback below
        if rg_result is not None and rg_result.returncode in (0, 1) \
                and not rg_result.timed_out:
            lines = rg_result.stdout.splitlines()[:max_results]
            return "\n".join(lines) if lines else "(no matches)"
        # Fallback: walk the tree with Python regex (works with no rg binary).
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise FileOpsError(f"invalid regex {pattern!r}: {exc}") from exc
        hits: list[str] = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for name in filenames:
                fpath = Path(dirpath) / name
                try:
                    text = fpath.read_text(encoding="utf-8", errors="strict")
                except (OSError, UnicodeDecodeError):
                    continue
                for lineno, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        rel = fpath.relative_to(self.root)
                        hits.append(f"{rel}:{lineno}:{line.strip()[:200]}")
                        if len(hits) >= max_results:
                            return "\n".join(hits)
        return "\n".join(hits) if hits else "(no matches)"

    # -- structure -------------------------------------------------------
    def ast_outline(self, rel_path: str) -> str:
        """Return classes/functions with signatures for one Python file."""
        path = self._resolve(rel_path)
        if not path.is_file():
            raise FileOpsError(f"not a file: {rel_path}")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            raise FileOpsError(f"syntax error in {rel_path}: {exc}") from exc
        out: list[str] = []

        def describe(node: ast.AST, indent: int) -> None:
            pad = "  " * indent
            if isinstance(node, ast.ClassDef):
                bases = ", ".join(ast.unparse(b) for b in node.bases)
                out.append(f"{pad}class {node.name}({bases}):  # line {node.lineno}")
                for child in node.body:
                    describe(child, indent + 1)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                signature = ast.unparse(node.args)
                out.append(f"{pad}{kind} {node.name}({signature})  # line {node.lineno}")

        for node in tree.body:
            describe(node, 0)
        return "\n".join(out) if out else "(no classes or functions)"

    # -- editing ---------------------------------------------------------
    def edit_file(self, rel_path: str, old_text: str, new_text: str) -> str:
        """Replace the single exact occurrence of *old_text* with *new_text*.

        Fails unless the anchor occurs exactly once, so ambiguous edits can
        never silently rewrite the wrong location.
        """
        path = self._resolve(rel_path)
        if not path.is_file():
            raise FileOpsError(f"not a file: {rel_path}")
        content = path.read_text(encoding="utf-8")
        count = content.count(old_text)
        if count == 0:
            raise FileOpsError(f"anchor text not found in {rel_path}")
        if count > 1:
            raise FileOpsError(
                f"anchor text occurs {count} times in {rel_path}; make it unique"
            )
        path.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"edited {rel_path} (1 replacement)"

    def write_file(self, rel_path: str, content: str) -> str:
        """Create a new file; refuses to overwrite an existing one."""
        path = self._resolve(rel_path)
        if path.exists():
            raise FileOpsError(f"file already exists (use edit_file): {rel_path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"wrote {rel_path} ({len(content)} chars)"
