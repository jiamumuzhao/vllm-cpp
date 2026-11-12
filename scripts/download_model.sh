#!/bin/bash
set -euo pipefail
MODEL="${1:-meta-llama/Llama-3.2-3B-Instruct}"
OUTDIR="${2:-./models}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Auto-activate project venv if it exists
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    export PATH="${PROJECT_DIR}/.venv/bin:$PATH"
fi

echo "=== vLLM-CPP Model Download ==="
echo "  Model:  ${MODEL}"
echo "  Target: ${OUTDIR}"

MODEL_NAME=$(basename "${MODEL}")
TARGET="${OUTDIR}/${MODEL_NAME}"

if [ -d "${TARGET}" ] && [ "$(ls -A "${TARGET}" 2>/dev/null)" ]; then
    echo "Model already exists at ${TARGET}"
    exit 0
fi

mkdir -p "${TARGET}"

# Strategy 1: hf CLI (new, recommended)
if command -v hf &>/dev/null; then
    echo "Using hf CLI..."
    hf download "${MODEL}" --local-dir "${TARGET}"
    echo "Model downloaded to ${TARGET}"
    exit 0
fi

# Strategy 2: legacy huggingface-cli
if command -v huggingface-cli &>/dev/null; then
    echo "Using huggingface-cli..."
    huggingface-cli download "${MODEL}" --local-dir "${TARGET}" --local-dir-use-symlinks False
    echo "Model downloaded to ${TARGET}"
    exit 0
fi

# Strategy 3: Python huggingface_hub
if command -v python3 &>/dev/null; then
    if python3 -c "import huggingface_hub" 2>/dev/null; then
        echo "Using Python huggingface_hub..."
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL}', local_dir='${TARGET}')
"
        echo "Model downloaded to ${TARGET}"
        exit 0
    fi
fi

echo ""
echo "ERROR: No download tool found. Install one of:"
echo "  pip install huggingface_hub[hf_transfer]   (recommended)"
echo "  pip install huggingface_hub"
echo ""
echo "Manually download from: https://huggingface.co/${MODEL}"
exit 1
