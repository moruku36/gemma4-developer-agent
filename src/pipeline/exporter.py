"""Assemble the offline Kaggle submission bundle.

Layout produced (uploaded to Kaggle as one or more datasets, then attached
to the submission notebook)::

    bundle/
      manifest.json          # file list + sha256 checksums
      notebook.ipynb         # copy of submission/notebook_template.ipynb
      code/                  # src/, configs/, scripts/, pyproject.toml
      models/gemma-4/        # weights + tokenizer (copied, never symlinked)
      wheels/                # pip wheels for offline install
      tasks/tasks.json       # task definitions (+ repos referenced by path)

Kaggle datasets do not preserve symlinks reliably, so model weights and
wheels are copied for real. The manifest lets the notebook verify the
bundle before spending GPU time.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_COPY_TREES = ("src", "configs", "scripts")
_COPY_FILES = ("pyproject.toml", "README.md")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src, dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )


def export_bundle(
    output_dir: str | Path,
    model_dir: str | Path | None = None,
    wheels_dir: str | Path | None = None,
    tasks_file: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict:
    """Build the bundle under *output_dir* and return its manifest."""
    root = Path(repo_root) if repo_root else REPO_ROOT
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    code_dir = out / "code"
    for tree in _COPY_TREES:
        _copy_tree(root / tree, code_dir / tree)
    for name in _COPY_FILES:
        shutil.copy2(root / name, code_dir / name)

    shutil.copy2(
        root / "submission" / "notebook_template.ipynb", out / "notebook.ipynb"
    )

    if model_dir is not None:
        _copy_tree(Path(model_dir), out / "models" / "gemma-4")
    else:
        placeholder = out / "models" / "gemma-4"
        placeholder.mkdir(parents=True, exist_ok=True)
        (placeholder / "PUT_WEIGHTS_HERE.txt").write_text(
            "Copy the Gemma 4 weights + tokenizer files into this directory "
            "(see scripts/setup_offline_deps.sh).\n", encoding="utf-8",
        )

    if wheels_dir is not None:
        _copy_tree(Path(wheels_dir), out / "wheels")
    else:
        (out / "wheels").mkdir(parents=True, exist_ok=True)

    if tasks_file is not None:
        tasks_dir = out / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tasks_file, tasks_dir / "tasks.json")

    manifest: dict = {"format": 1, "files": []}
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        if path.name == "manifest.json":
            continue
        manifest["files"].append({
            "path": str(path.relative_to(out)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
