#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

mkdir -p \
  /d/AI-Caches/tmp \
  /d/AI-Caches/torch \
  /d/AI-Caches/cuda \
  /d/AI-Caches/triton \
  /d/AI-Caches/torchinductor \
  /d/AI-Caches/xdg \
  /d/AI-Caches/huggingface/xet \
  outputs

export OLLAMA_MODELS='D:\AI-Caches\ollama\models'
export HF_HOME='D:\AI-Caches\huggingface'
export HF_HUB_CACHE='D:\AI-Caches\huggingface\hub'
export HF_XET_CACHE='D:\AI-Caches\huggingface\xet'
export PIP_CACHE_DIR='D:\AI-Caches\pip'
export TEMP='D:\AI-Caches\tmp'
export TMP='D:\AI-Caches\tmp'
export TMPDIR='D:\AI-Caches\tmp'
export TORCH_HOME='D:\AI-Caches\torch'
export CUDA_CACHE_PATH='D:\AI-Caches\cuda'
export TRITON_CACHE_DIR='D:\AI-Caches\triton'
export TORCHINDUCTOR_CACHE_DIR='D:\AI-Caches\torchinductor'
export XDG_CACHE_HOME='D:\AI-Caches\xdg'

# All required Hugging Face assets are prefetched. Fail instead of downloading
# anything unexpectedly during the long experiment.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHON="$PROJECT_DIR/.venv/Scripts/python.exe"
LOG_FILE="$PROJECT_DIR/outputs/midterm.log"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing D: virtual environment: $PYTHON" >&2
  exit 1
fi

if [[ "${1:-}" == "--check" ]]; then
  variables=(
    OLLAMA_MODELS HF_HOME HF_HUB_CACHE HF_XET_CACHE PIP_CACHE_DIR
    TEMP TMP TMPDIR TORCH_HOME CUDA_CACHE_PATH TRITON_CACHE_DIR
    TORCHINDUCTOR_CACHE_DIR XDG_CACHE_HOME
  )
  status=0
  for variable in "${variables[@]}"; do
    value="${!variable:-}"
    printf '%s=%s\n' "$variable" "$value"
    if [[ "$value" =~ ^[Cc]: ]]; then
      echo "ERROR: $variable still points to C:" >&2
      status=1
    fi
  done
  "$PYTHON" -c \
    "import tempfile; print('PYTHON_TEMP=' + tempfile.gettempdir())"
  if [[ "$status" -eq 0 ]]; then
    echo "OK: every experiment cache and temporary path points away from C:."
  fi
  exit "$status"
fi

if [[ "${1:-}" == "--resume" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "Usage: ./run_midterm.sh --resume outputs/RUN_DIRECTORY" >&2
    exit 2
  fi
  RUN_DIR="$2"
  echo "Resuming: $RUN_DIR"
  "$PYTHON" -u run_attack_defense_matrix.py \
    --output-dir "$RUN_DIR" \
    --resume \
    2>&1 | tee -a "$LOG_FILE"
else
  RUN_DIR="outputs/midterm-50-$(date +%Y%m%d-%H%M%S)"
  echo "Starting: $RUN_DIR"
  "$PYTHON" -u run_attack_defense_matrix.py \
    --output-dir "$RUN_DIR" \
    2>&1 | tee -a "$LOG_FILE"
fi
