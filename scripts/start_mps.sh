#!/usr/bin/env bash
# scripts/start_mps.sh — Start CUDA MPS daemon for PersonaPlex GPU partitioning
#
# MPS partitions GPU SMs at the hardware level so that PersonaPlex pipeline
# and brain can run concurrently without contention.  Each container gets a
# dedicated slice of GPU streaming multiprocessors (SMs):
#   - triton (PP):  80% of SMs  → ~75ms frames (< 80ms budget)
#   - brain:        20% of SMs  → ~3 tok/s (background context, speed not critical)
#
# Run this on the HOST before docker compose up:
#   sudo bash scripts/start_mps.sh
#
# To stop:
#   echo quit | sudo nvidia-cuda-mps-control
#   sudo nvidia-smi -i 0 -c DEFAULT

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log

mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

echo "=== PersonaPlex MPS Setup ==="

# Check if MPS daemon is already running
if echo "get_server_list" | nvidia-cuda-mps-control 2>/dev/null; then
    echo "MPS daemon already running. Restarting..."
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    sleep 1
fi

# Set GPU to EXCLUSIVE_PROCESS mode (required for MPS)
echo "Setting GPU 0 to EXCLUSIVE_PROCESS mode..."
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS

# Start MPS daemon
echo "Starting MPS daemon..."
nvidia-cuda-mps-control -d

# Verify
sleep 1
if echo "get_server_list" | nvidia-cuda-mps-control 2>/dev/null; then
    echo "✅ MPS daemon running"
    echo "   Pipe directory: $CUDA_MPS_PIPE_DIRECTORY"
    echo "   Log directory:  $CUDA_MPS_LOG_DIRECTORY"
    echo ""
    echo "Now start containers with:"
    echo "   cd docker && docker compose up -d"
else
    echo "❌ MPS daemon failed to start"
    exit 1
fi
