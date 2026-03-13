#!/usr/bin/env bash
# scripts/start_triton.sh — Start Triton Inference Server with the model repository
# Run inside the Triton Docker container or from docker-compose.
#
# Modes:
#   LOAD_BRAIN=1         — load both personaplex_pipeline + brain (single-container)
#   BRAIN_ONLY=1         — load ONLY brain (MPS split: separate brain container)
#   (default)            — load only personaplex_pipeline

set -euo pipefail

REPO_DIR="${MODEL_REPO:-$(cd "$(dirname "$0")/.." && pwd)/model_repository}"
LOG_VERBOSE="${LOG_VERBOSE:-0}"
GRPC_PORT="${GRPC_PORT:-8001}"
HTTP_PORT="${HTTP_PORT:-8000}"
METRICS_PORT="${METRICS_PORT:-8002}"
LOAD_BRAIN="${LOAD_BRAIN:-0}"
BRAIN_ONLY="${BRAIN_ONLY:-0}"

echo "Starting Triton Inference Server"
echo "  Model repository : $REPO_DIR"
echo "  gRPC port        : $GRPC_PORT"
echo "  HTTP port        : $HTTP_PORT"
echo "  Metrics port     : $METRICS_PORT"
echo "  Load Brain       : $LOAD_BRAIN"
echo "  Brain Only       : $BRAIN_ONLY"

# Build --load-model flags
LOAD_FLAGS=""
if [ "${BRAIN_ONLY}" = "1" ]; then
    LOAD_FLAGS="--load-model=brain"
    echo "  Mode: BRAIN ONLY (MPS isolated)"
else
    LOAD_FLAGS="--load-model=personaplex_pipeline"
    if [ "${LOAD_BRAIN}" = "1" ]; then
        LOAD_FLAGS="${LOAD_FLAGS} --load-model=brain"
        echo "  Mode: PP + Brain (single container)"
    else
        echo "  Mode: PP only"
    fi
fi

exec tritonserver \
    --model-repository="${REPO_DIR}" \
    --grpc-port="${GRPC_PORT}" \
    --http-port="${HTTP_PORT}" \
    --metrics-port="${METRICS_PORT}" \
    --log-verbose="${LOG_VERBOSE}" \
    --model-control-mode=explicit \
    ${LOAD_FLAGS} \
    --strict-readiness=true \
    --exit-on-error=false \
    --backend-config=python,shm-default-byte-size=134217728 \
    --rate-limit=execution_count \
    "$@"
