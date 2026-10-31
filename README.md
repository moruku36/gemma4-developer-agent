# Gemma 4 Developer Agent — Offline Coding Agent Harness

Execution infrastructure for the **Gemma 4 Developer Agent Competition**
(Google DeepMind, on Kaggle): an autonomous coding agent that runs a
Gemma 4 model **fully offline** on a Kaggle notebook GPU instance
(L4x4-class), explores an unseen Python repository, and produces a patch
(diff) that is scored SWE-bench style — hidden tests, Pass/Fail.

Drop in fine-tuned or baseline Gemma 4 weights and this repo gives you:
local evaluation on public tasks, deterministic run limits, patch
validation, and one-command packaging into the Kaggle submission bundle.

## Team roles

| Role | Owner | Responsibility |
| --- | --- | --- |
| Infra / Agent Systems Lead | This repository | Offline execution harness, vLLM serving, agent loop, file/edit tools, syntax & test sandbox, Kaggle submission pipeline |
| ML / Modeling Lead | Teammate | Gemma 4 post-training (PEFT/LoRA), prompt tuning, inference optimization; delivers the checkpoint this harness serves |

The interface between the two sides is deliberately thin: a weights
directory (weights + tokenizer) and the config files in `configs/`.

## Architecture

```text
Kaggle notebook (GPU, internet OFF, <= 12 h total)
│
├─ submission/notebook_template.ipynb   entry point: locate bundle ->
│                                       start server -> run -> predictions
├─ src/engine/vllm_server.py            vLLM subprocess lifecycle
│                                       (spawn, /health polling, clean stop)
├─ src/engine/client.py                 OpenAI-compatible local client
│                                       (standard library only)
├─ src/agent/core.py                    ReAct / plan-and-solve loop with hard
│                                       caps: steps, tokens, wall clock
├─ src/agent/prompts.py                 system prompt + strict JSON tool
│                                       protocol + few-shot examples
├─ src/agent/memory.py                  context budget + tool-output pruning
├─ src/tools/file_ops.py                read / edit / search / AST outline,
│                                       confined to the task repository
├─ src/tools/patch.py                   git diff generation; validation =
│                                       git apply --check on a pristine clone
│                                       + compile() of touched .py files
├─ src/tools/execution.py               subprocess runs with timeout,
│                                       process-group kill, bounded output
└─ src/pipeline/runner.py               batch runner: disposable repo copy
                                        per task, one global time budget
```

Design choices worth knowing:

- **Text tool protocol, not native function calling.** The model replies
  with exactly one JSON object per turn (a tool call or `finish`). This
  works with any checkpoint, fine-tuned or not, and is fully loggable.
- **Deterministic guardrails.** Step/token/time budgets live in the
  harness (`configs/default_agent.yaml`), not in prompts.
- **No patch without validation.** A task only counts as solved when its
  diff applies cleanly at the original HEAD and every touched Python
  file compiles.
- **Prompt-injection posture.** Repository and issue text is treated as
  data: the system prompt forbids following instructions embedded in
  files, and file tools cannot escape the task repository root.

## Offline constraints

Scoring runs with **no internet**. Everything must be bundled in advance:

- model weights + tokenizer (`models/gemma-4/`, via Kaggle dataset)
- pip wheels (`wheels/`, installed with `pip install --no-index`)
- this code + configs (`code/` in the bundle)

`scripts/setup_offline_deps.sh` is the only step that uses the network,
and it runs during preparation, never on Kaggle.

## 12-hour budget breakdown (default config)

| Phase | Budget |
| --- | --- |
| Notebook setup + offline install | ~10 min |
| vLLM startup (model load, L4x4) | up to 15 min (health timeout) |
| Agent runs over all tasks | ~11 h (global cap 41,400 s) |
| Per-task caps | 60 steps / 400k tokens / 30 min each |
| Packaging predictions + reserve | 15 min reserve kept unspent |

## Quick start (local, no GPU needed for the smoke test)

```bash
pip install -e ".[dev]"
pytest                                   # unit tests: fake LLM client

# Harness end-to-end smoke test (scripted client, two toy bug-fix tasks):
python scripts/run_local_eval.py --demo

# Real local evaluation against a running vLLM server:
python -m vllm.entrypoints.openai.api_server --model models/gemma-4 \
    --tensor-parallel-size 4 --port 8000   # or your own serving setup
python scripts/run_local_eval.py --tasks tasks.json --output artifacts/local-eval
```

`tasks.json` format:

```json
[{"task_id": "demo-1", "repo_path": "/path/to/checkout",
  "issue_text": "describe the bug...", "test_cmd": "python3 -m pytest -q"}]
```

## Building the Kaggle submission

```bash
# 1. Fetch wheels + weights (internet ON, preparation only)
MODEL_ID=<hf-repo-of-the-fine-tuned-checkpoint> bash scripts/setup_offline_deps.sh
#    ...or MODEL_LOCAL_DIR=/path/to/checkpoint for a local fine-tune output

# 2. Assemble the bundle (code + notebook + weights + wheels + tasks)
python scripts/build_submission.py --output artifacts/bundle \
    --model-dir models/gemma-4 --wheels-dir wheels --tasks tasks.json

# 3. Upload artifacts/bundle as Kaggle dataset(s), create a notebook from
#    submission/notebook_template.ipynb, attach the dataset, enable GPU,
#    keep internet OFF, and run. Output: /kaggle/working/predictions.jsonl
```

## Repository layout

```text
configs/     agent + vLLM configuration (the two tuning surfaces)
src/agent/   loop, prompts, memory        src/engine/  serving + client
src/tools/   file ops, patch, execution   src/pipeline/ runner + exporter
scripts/     offline prep, local eval, bundle build
tests/       pytest unit tests (no GPU / vLLM required)
submission/  Kaggle notebook template
```

## Status

Scaffold v0.1: harness, validation, packaging, and tests are in place and
green. Next iterations: richer repo navigation (symbol graph), retry with
failure-feedback, and checkpoint swaps from the ML lead.
