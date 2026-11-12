#!/bin/bash
# Ubuntu 26.04 environment setup for vLLM-CPP development
set -euo pipefail

echo "=== vLLM-CPP Environment Setup ==="
echo "Installing build dependencies..."
sudo apt update
sudo apt install -y build-essential cmake ninja-build git ccache
sudo apt install -y libssl-dev libcurl4-openssl-dev

echo "Verifying CUDA toolkit..."
nvcc --version || echo "WARNING: nvcc not found. Install the CUDA toolkit."

echo "Verifying NVIDIA driver..."
nvidia-smi || echo "WARNING: nvidia-smi failed. Check driver installation."
echo "Setup complete."
