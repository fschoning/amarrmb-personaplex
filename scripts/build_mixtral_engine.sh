#!/usr/bin/env bash
# scripts/build_mixtral_engine.sh
# One-time AOT compilation of Mixtral-8x7B NVFP4 → TensorRT-LLM engine
# Run INSIDE the TRT-LLM container on the DGX Spark.
#
# Usage:
#   docker run --rm --gpus all \
#     -v /mnt/models:/mnt/models \
#     nvcr.io/nvidia/tritonserver:25.04-trtllm-python-py3 \
#     bash /mnt/models/build_mixtral_engine.sh
#
# Or if trtllm is installed in the current environment:
#   bash scripts/build_mixtral_engine.sh

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
CKPT_DIR="${MIXTRAL_CKPT:-/mnt/models/Mixtral-8x7B-Instruct-v0.1-NVFP4}"
ENGINE_DIR="${MIXTRAL_ENGINE:-/mnt/models/mixtral-engine}"
CONVERTED_DIR="${ENGINE_DIR}/checkpoint"

echo "══════════════════════════════════════════════════════"
echo "  Mixtral 8x7B TensorRT-LLM Engine Build"
echo "══════════════════════════════════════════════════════"
echo "  Checkpoint : $CKPT_DIR"
echo "  Engine out : $ENGINE_DIR"
echo ""

# ── Verify inputs ─────────────────────────────────────────────────────────────
if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $CKPT_DIR"
    echo "Download with:"
    echo "  HF_HOME=/mnt/models/hf-cache huggingface-cli download \\"
    echo "    josephdowling10/Mixtral-8x7B-Instruct-v0.1-NVFP4 \\"
    echo "    --local-dir $CKPT_DIR --local-dir-use-symlinks False"
    exit 1
fi

# ── Check trtllm-build availability ───────────────────────────────────────────
if ! command -v trtllm-build &>/dev/null; then
    echo "ERROR: trtllm-build not found."
    echo ""
    echo "Install TensorRT-LLM or run inside the TRT-LLM container:"
    echo "  docker run --rm --gpus all \\"
    echo "    -v /mnt/models:/mnt/models \\"
    echo "    nvcr.io/nvidia/tritonserver:25.04-trtllm-python-py3 \\"
    echo "    bash /path/to/build_mixtral_engine.sh"
    exit 1
fi

echo "✓ trtllm-build found: $(which trtllm-build)"
echo ""

# ── Step 1: Convert NVFP4 checkpoint to TRT-LLM checkpoint format ─────────────
mkdir -p "$CONVERTED_DIR"

# Check if the checkpoint is already in TRT-LLM format (has config.json)
if [ -f "$CKPT_DIR/config.json" ] && grep -q "builder_config\|quantization" "$CKPT_DIR/config.json" 2>/dev/null; then
    echo "Checkpoint appears to be in TRT-LLM format — skipping conversion."
    CONVERTED_DIR="$CKPT_DIR"
else
    echo "Step 1/2: Converting NVFP4 checkpoint to TRT-LLM format..."
    TRTLLM_ROOT=$(python3 -c "import tensorrt_llm; import os; print(os.path.dirname(tensorrt_llm.__file__))" 2>/dev/null || echo "")
    if [ -z "$TRTLLM_ROOT" ]; then
        echo "ERROR: tensorrt_llm Python package not found"
        exit 1
    fi

    python3 "$TRTLLM_ROOT/examples/mixtral/convert_checkpoint.py" \
        --model_dir "$CKPT_DIR" \
        --output_dir "$CONVERTED_DIR" \
        --dtype bfloat16 \
        --use_fp4_weights \
        --tp_size 1 \
        --pp_size 1
    echo "✓ Conversion done → $CONVERTED_DIR"
fi

# ── Step 2: Build TRT-LLM engine ──────────────────────────────────────────────
echo ""
echo "Step 2/2: Building TRT-LLM engine (this takes 10-30 minutes)..."
mkdir -p "$ENGINE_DIR"

trtllm-build \
    --checkpoint_dir "$CONVERTED_DIR" \
    --output_dir "$ENGINE_DIR" \
    --max_batch_size 4 \
    --max_input_len 4096 \
    --max_seq_len 6144 \
    --gemm_plugin bfloat16 \
    --gpt_attention_plugin bfloat16 \
    --use_fp8_context_fmha enable \
    --workers 1

echo ""
echo "══════════════════════════════════════════════════════"
echo "  ✅ Engine build complete!"
echo "  Engine saved to: $ENGINE_DIR"
echo "  Files:"
ls -lh "$ENGINE_DIR/"
echo ""
echo "  Next: restart Triton with mixtral_brain model loaded."
echo "══════════════════════════════════════════════════════"
