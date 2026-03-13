#!/usr/bin/env bash
# scripts/stop_mps.sh — Stop CUDA MPS daemon and restore GPU to default mode
set -euo pipefail

echo "Stopping MPS daemon..."
echo quit | nvidia-cuda-mps-control 2>/dev/null || echo "(not running)"

echo "Restoring GPU 0 to DEFAULT compute mode..."
nvidia-smi -i 0 -c DEFAULT

echo "✅ MPS stopped. GPU restored to default mode."
