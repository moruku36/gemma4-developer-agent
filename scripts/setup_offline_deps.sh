#!/usr/bin/env bash
# Fetch everything the offline scoring run needs, while we still HAVE
# internet (preparation time only — nothing here runs on Kaggle).
#
# Produces:
#   wheels/          pip wheels installed with --no-index on the notebook
#   models/gemma-4/  model weights + tokenizer (baseline or fine-tuned)
#
# Environment overrides:
#   MODEL_ID          HF repo id to download (default: set by the ML lead to
#                     the fine-tuned checkpoint, e.g. your-org/gemma-4-agent)
#   MODEL_LOCAL_DIR   if set, copy weights from this local dir instead
#   DOWNLOAD_VLLM=1   also download the vllm wheel (large; only when the
#                     Kaggle image does not already provide it)
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p wheels models/gemma-4

echo "==> Downloading Python wheels into wheels/"
pip download -d wheels/ "pyyaml>=6.0" "openai>=1.30"
if [[ "${DOWNLOAD_VLLM:-0}" == "1" ]]; then
  echo "==> Downloading vLLM wheel (large)"
  pip download -d wheels/ vllm
fi

if [[ -n "${MODEL_LOCAL_DIR:-}" ]]; then
  echo "==> Copying model weights from ${MODEL_LOCAL_DIR}"
  cp -r "${MODEL_LOCAL_DIR}/." models/gemma-4/
elif [[ -n "${MODEL_ID:-}" ]]; then
  echo "==> Downloading model ${MODEL_ID} from Hugging Face"
  huggingface-cli download "${MODEL_ID}" --local-dir models/gemma-4
else
  echo "!! Neither MODEL_LOCAL_DIR nor MODEL_ID is set — skipping weights."
  echo "   Set one of them (the ML lead owns the checkpoint choice), e.g.:"
  echo "   MODEL_ID=<hf-repo-id> bash scripts/setup_offline_deps.sh"
fi

echo "==> Bundle payload sizes:"
du -sh wheels models/gemma-4 || true
echo "Done. Next: python scripts/build_submission.py --output artifacts/bundle"
