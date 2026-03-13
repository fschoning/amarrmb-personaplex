#!/usr/bin/env bash
# scripts/start_triton.sh — Start Triton Inference Server with the model repository
# Run inside the Triton Docker container or from docker-compose.

set -euo pipefail

REPO_DIR="${MODEL_REPO:-$(cd "$(dirname "$0")/.." && pwd)/model_repository}"
LOG_VERBOSE="${LOG_VERBOSE:-0}"
GRPC_PORT="${GRPC_PORT:-8001}"
HTTP_PORT="${HTTP_PORT:-8000}"
METRICS_PORT="${METRICS_PORT:-8002}"
LOAD_BRAIN="${LOAD_BRAIN:-0}"

echo "Starting Triton Inference Server"
echo "  Model repository : $REPO_DIR"
echo "  gRPC port        : $GRPC_PORT"
echo "  HTTP port        : $HTTP_PORT"
echo "  Metrics port     : $METRICS_PORT"
echo "  Load Brain       : $LOAD_BRAIN"

# Conditionally add brain if LOAD_BRAIN=1
BRAIN_FLAG=""
if [ "${LOAD_BRAIN}" = "1" ]; then
    BRAIN_FLAG="--load-model=brain"
fi

exec tritonserver \
    --model-repository="${REPO_DIR}" \
    --grpc-port="${GRPC_PORT}" \
    --http-port="${HTTP_PORT}" \
    --metrics-port="${METRICS_PORT}" \
    --log-verbose="${LOG_VERBOSE}" \
    --model-control-mode=explicit \
    --load-model=personaplex_pipeline \
    ${BRAIN_FLAG} \
    --strict-readiness=true \
    --exit-on-error=false \
    --backend-config=python,shm-default-byte-size=134217728 \
    "$@"
