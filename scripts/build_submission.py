#!/usr/bin/env python3
"""Build the Kaggle submission bundle (and optionally a zip of it).

Usage:
  python scripts/build_submission.py --output artifacts/bundle \
      --model-dir models/gemma-4 --wheels-dir wheels --tasks tasks.json [--zip]

Upload the bundle contents as Kaggle dataset(s), attach them to a notebook
created from submission/notebook_template.ipynb, and run it with GPU +
internet OFF.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.exporter import export_bundle  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="weights dir to embed (default: placeholder)")
    parser.add_argument("--wheels-dir", type=Path, default=None)
    parser.add_argument("--tasks", type=Path, default=None, help="tasks.json to embed")
    parser.add_argument("--zip", action="store_true", help="also write <output>.zip")
    args = parser.parse_args(argv)

    manifest = export_bundle(
        args.output,
        model_dir=args.model_dir,
        wheels_dir=args.wheels_dir,
        tasks_file=args.tasks,
        repo_root=REPO_ROOT,
    )
    total_bytes = sum(f["bytes"] for f in manifest["files"])
    print(f"bundle: {args.output} ({len(manifest['files'])} files, "
          f"{total_bytes / 1e9:.2f} GB)")
    if args.zip:
        archive = shutil.make_archive(str(args.output), "zip", args.output)
        print(f"zip:    {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
