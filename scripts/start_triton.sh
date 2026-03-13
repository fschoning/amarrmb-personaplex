#!/usr/bin/env bash
# scripts/start_triton.sh — Start Triton Inference Server with the model repository
# Run inside the Triton Docker container or from docker-compose.

set -euo pipefail

REPO_DIR="${MODEL_REPO:-$(cd "$(dirname "$0")/.." && pwd)/model_repository}"
LOG_VERBOSE="${LOG_VERBOSE:-0}"
GRPC_PORT="${GRPC_PORT:-8001}"
HTTP_PORT="${HTTP_PORT:-8000}"
METRICS_PORT="${METRICS_PORT:-8002}"
LOAD_MIXTRAL="${LOAD_MIXTRAL:-0}"

echo "Starting Triton Inference Server"
echo "  Model repository : $REPO_DIR"
echo "  gRPC port        : $GRPC_PORT"
echo "  HTTP port        : $HTTP_PORT"
echo "  Metrics port     : $METRICS_PORT"
echo "  Load Mixtral     : $LOAD_MIXTRAL"

# Conditionally add mixtral_brain if LOAD_MIXTRAL=1
MIXTRAL_FLAG=""
if [ "${LOAD_MIXTRAL}" = "1" ]; then
    MIXTRAL_FLAG="--load-model=mixtral_brain"
fi

exec tritonserver \
    --model-repository="${REPO_DIR}" \
    --grpc-port="${GRPC_PORT}" \
    --http-port="${HTTP_PORT}" \
    --metrics-port="${METRICS_PORT}" \
    --log-verbose="${LOG_VERBOSE}" \
    --model-control-mode=explicit \
    --load-model=personaplex_pipeline \
    ${MIXTRAL_FLAG} \
    --strict-readiness=true \
    --exit-on-error=false \
    --backend-config=python,shm-default-byte-size=134217728 \
    "$@"
