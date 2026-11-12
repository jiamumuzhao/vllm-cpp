#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODEL_PATH="${MODEL_PATH:-./models/Qwen2.5-0.5B-Instruct}"
MODEL_ARCH="${MODEL_ARCH:-qwen}"
QUANT_MODE="${QUANT_MODE:-}"
PORT="${PORT:-8000}"

if command -v ss >/dev/null 2>&1; then
    PORT_LISTENER="$(ss -H -ltnp "( sport = :${PORT} )" 2>/dev/null || true)"
    if [ -n "${PORT_LISTENER}" ]; then
        echo "Port ${PORT} is already in use. Stop the existing server or set PORT to another port." >&2
        echo "${PORT_LISTENER}" >&2
        exit 1
    fi
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "Model directory not found: ${MODEL_PATH}" >&2
    echo "Set MODEL_PATH to a downloaded Qwen model directory." >&2
    exit 1
fi

BUILD_DIR="${PROJECT_DIR}/build"

if [ ! -f "${BUILD_DIR}/bin/vllm_server" ]; then
    echo "Binary not found. Building..."
    cmake -B "${BUILD_DIR}" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_CUDA_ARCHITECTURES=75 \
        -DVLLM_ENABLE_TESTS=OFF \
        -DVLLM_ENABLE_BENCHMARKS=OFF \
        -DVLLM_NETWORK_DEPS=OFF
    cmake --build "${BUILD_DIR}" -j"$(nproc)"
fi

ARGS=(
    --model "${MODEL_PATH}"
    --arch "${MODEL_ARCH}"
    --port "${PORT}"
)
[ -n "${QUANT_MODE}" ] && ARGS+=(--quantization "${QUANT_MODE}")

echo "=== Starting vLLM-CPP ==="
echo "  Model:  ${MODEL_PATH}"
echo "  Arch:   ${MODEL_ARCH}"
echo "  Quant:  ${QUANT_MODE:-none}"
echo "  Port:   ${PORT}"
echo "  Binary: ${BUILD_DIR}/bin/vllm_server"
echo

exec "${BUILD_DIR}/bin/vllm_server" "${ARGS[@]}"
