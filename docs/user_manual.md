# PersonaPlex v2 — User Manual

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Prerequisites](#2-prerequisites)
3. [Project Structure](#3-project-structure)
4. [Configuration](#4-configuration)
5. [Building on DGX Spark (Production)](#5-building-on-dgx-spark-production)
6. [Building & Testing Without DGX Spark](#6-building--testing-without-dgx-spark)
7. [Deploying to DGX Spark](#7-deploying-to-dgx-spark)
8. [API Reference](#8-api-reference)
9. [Operations & Monitoring](#9-operations--monitoring)
10. [Tuning & Performance](#10-tuning--performance)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. System Overview

PersonaPlex v2 replaces the monolithic Python server with a two-container architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                        DGX Spark                                │
│                                                                 │
│  ┌──────────────────┐        ┌──────────────────────────────┐   │
│  │  C++ Gateway     │ gRPC   │  Triton Inference Server     │   │
│  │  (WebSocket API) │───────►│                              │   │
│  │  Port 8998       │        │  mimi_encoder    (×6 inst.)  │   │
│  │                  │        │  personaplex_lm  (×6 inst.)  │   │
│  │  ~5 MB binary    │        │  mimi_decoder    (×6 inst.)  │   │
│  │  ~0.5 MB RAM/    │        │  lavasr_v2       (×1 inst.)  │   │
│  │   session        │        │                              │   │
│  └──────────────────┘        │  Port 8001 (gRPC)            │   │
│                              │  Port 8000 (HTTP health)     │   │
│                              │  Port 8002 (Prometheus)      │   │
│                              └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Gateway container** — ~50 MB runtime image. Handles all client WebSocket connections,
implements the OpenAI Realtime API, manages sessions, and proxies audio to Triton.

**Triton container** — ~15 GB image (due to PyTorch + model weights). Runs the 4-step GPU
inference pipeline: Mimi encode → PersonaPlex LM → Mimi decode → LavaSR v2 upsample.

---

## 2. Prerequisites

### For all environments

| Tool | Version | Purpose |
|------|---------|---------|
| Docker Engine | ≥ 24.0 | Container runtime |
| Docker Compose | ≥ 2.20 | Multi-container orchestration |
| Git | any | Source control |

### Additional for DGX Spark (production)

| Tool | Version | Purpose |
|------|---------|---------|
| NVIDIA Container Toolkit | ≥ 1.14 | GPU access from Docker |
| NVIDIA Driver | ≥ 560 | Blackwell SM 12.1 support |
| HuggingFace account | — | Access to model weights (if gated) |

### Additional for dev/test without GPU

| Tool | Version | Purpose |
|------|---------|---------|
| Python | ≥ 3.10 | Running unit tests, mock clients |
| wscat or websocat | any | CLI WebSocket testing |
| Node.js | ≥ 18 | If using OpenAI Node SDK for testing |

---

## 3. Project Structure

```
amarrmb-personaplex/
├── .env.example                  # Template for environment configuration
├── README-v2.md                  # Quick-start deployment guide
├── docs/
│   └── user_manual.md            # This file
│
├── docker/
│   ├── Dockerfile.triton         # Triton image (PyTorch + moshi + LavaSR)
│   ├── Dockerfile.gateway        # Gateway image (multi-stage C++ build)
│   └── docker-compose.yaml       # Full stack definition
│
├── gateway/                      # C++ gateway source
│   ├── CMakeLists.txt            # Build system (fetches deps via FetchContent)
│   └── src/
│       ├── main.cpp              # Entry point — uWebSockets server + session workers
│       ├── config.h              # Runtime config from env vars
│       ├── audio_utils.h/.cpp    # Base64 + PCM conversion
│       ├── protocol.h/.cpp       # OpenAI Realtime API parser/serialiser
│       ├── session.h/.cpp        # Ring buffer, Session, SessionManager
│       └── triton_client.h/.cpp  # Triton gRPC wrapper
│
├── model_repository/             # Triton model configs + Python backends
│   ├── personaplex_pipeline/
│   │   └── config.pbtxt          # Ensemble DAG (4-step pipeline)
│   ├── mimi_encoder/
│   │   ├── config.pbtxt          # Sequence batcher, 6 GPU instances
│   │   └── 1/model.py            # MimiModel.encode() wrapper
│   ├── personaplex_lm/
│   │   ├── config.pbtxt          # Sequence batcher, 6 GPU instances
│   │   └── 1/model.py            # LMGen.step() + system prompt conditioning
│   ├── mimi_decoder/
│   │   ├── config.pbtxt          # Sequence batcher, 6 GPU instances
│   │   └── 1/model.py            # MimiModel.decode() wrapper
│   └── lavasr_v2/
│       ├── config.pbtxt          # Stateless upsampler, 1 GPU instance
│       └── 1/model.py            # LavaSR v2 24kHz→48kHz
│
├── scripts/
│   ├── start_triton.sh           # Triton startup wrapper
│   ├── export_lavasr.py          # PyTorch → ONNX → TensorRT export
│   └── export_mimi.py            # Phase 2 stub (not yet functional)
│
└── moshi/                        # Existing moshi Python package (unchanged)
    ├── pyproject.toml
    ├── requirements.txt
    └── moshi/                    # MimiModel, LMGen, loaders, etc.
```

---

## 4. Configuration

All runtime configuration is driven by environment variables. Copy `.env.example` to `.env`
and edit before starting the stack:

```bash
cp .env.example .env
nano .env
```

### Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | *(empty)* | HuggingFace access token (required if model is gated) |
| `HF_REPO` | *(empty → kyutai/moshiko-pytorch-bf16)* | HuggingFace repo for Moshi weights |
| `MAX_SESSIONS` | `6` | Max concurrent sessions. **Must match** `instance_group.count` in the three `config.pbtxt` files |
| `PERSONAPLEX_FP8` | `1` | Enable FP8 quantisation (Blackwell only). Set `0` for Ampere/Hopper |
| `LM_TEMPERATURE` | `0.8` | LM sampling temperature |
| `LM_TOP_K` | `250` | LM top-k sampling |
| `LAVASR_HF_REPO` | `declinator/lava-sr-v2` | HuggingFace repo for LavaSR v2 |
| `WS_PORT` | `8998` | Gateway WebSocket port |
| `SESSION_TIMEOUT_S` | `300` | Idle session timeout (seconds) |
| `SSL_CERT` | *(empty)* | Path to TLS certificate (leave blank for no TLS) |
| `SSL_KEY` | *(empty)* | Path to TLS private key |
| `SSL_DIR` | `./ssl` | Directory mounted into gateway for TLS files |
| `GRPC_PORT` | `8001` | Triton gRPC port |
| `HTTP_PORT` | `8000` | Triton HTTP health check port |
| `METRICS_PORT` | `8002` | Triton Prometheus metrics port |

### Changing Concurrent Session Count

If you change `MAX_SESSIONS`, you **must also** edit three `config.pbtxt` files:

- `model_repository/mimi_encoder/config.pbtxt`
- `model_repository/personaplex_lm/config.pbtxt`
- `model_repository/mimi_decoder/config.pbtxt`

In each, find the `instance_group` block and set `count` to match:

```protobuf
instance_group [
  {
    count: 6    # ← must match MAX_SESSIONS
    kind: KIND_GPU
    gpus: [0]
  }
]
```

---

## 5. Building on DGX Spark (Production)

This is the standard path when you have access to the DGX Spark.

### Step 1: Prepare Environment

```bash
# SSH to DGX Spark
ssh user@dgx-spark

# Pull latest code
git clone <repo-url> amarrmb-personaplex
cd amarrmb-personaplex

# Configure
cp .env.example .env
nano .env   # Set HF_TOKEN, verify MAX_SESSIONS=6
```

### Step 2: Build Containers

```bash
cd docker

# Build both containers (first build: 15-30 min — downloads 15 GB base images)
docker compose build
```

> **Note:** Ensure you have at least 40 GB free disk space for Docker images and build layers.

### Step 3: Start the Stack

```bash
docker compose up -d
```

This starts:
1. **Triton** — loads model weights from HuggingFace (first run: 3-10 min download,
   2-5 min GPU load). Health check polls every 15s, up to 12 retries, with 120s start period.
2. **Gateway** — starts only after Triton passes its health check.

### Step 4: Verify

```bash
# Check Triton health
curl http://localhost:8000/v2/health/ready
# → {"ready":true}

# Check Triton model status
curl http://localhost:8000/v2/models/personaplex_pipeline

# Check gateway logs
docker compose logs gateway
# → "Listening on ws://0.0.0.0:8998/v1/realtime"
```

### Step 5: (Optional) Export LavaSR v2 to TensorRT

Replaces the Python LavaSR backend with a native TensorRT engine for lower latency.

```bash
# Run export inside the Triton container (requires GPU)
docker run --gpus all --rm \
  -v $(pwd)/model_repository:/models \
  personaplex-triton:latest \
  python /app/scripts/export_lavasr.py --out-dir /models/lavasr_v2/1

# Activate the TensorRT backend
sed -i 's/backend: "python"/backend: "tensorrt"/' \
  model_repository/lavasr_v2/config.pbtxt
mv model_repository/lavasr_v2/1/model.py \
   model_repository/lavasr_v2/1/model.py.bak

# Restart Triton to pick up the change
docker compose restart triton
```

---

## 6. Building & Testing Without DGX Spark

> **Architecture note:** The DGX Spark uses an ARM64 CPU (NVIDIA Grace / Neoverse V2).
> C++ binaries compiled on an x86 machine will not run on the Spark. Build the final
> production image on the Spark itself. You can, however, fully validate the Python
> backends and most of the C++ logic from an x86 machine using the strategies below.

### Strategy A: Build the Gateway Only (No GPU needed)

```bash
# Build via Docker — compiles the C++ gateway for the host platform
cd amarrmb-personaplex
docker build -f docker/Dockerfile.gateway -t personaplex-gateway:test .

# Run standalone (will fail to reach Triton gracefully)
docker run --rm -p 8998:8998 \
  -e TRITON_GRPC_URL=localhost:8001 \
  -e MAX_SESSIONS=2 \
  personaplex-gateway:test

# Test in another terminal with wscat (npm install -g wscat)
wscat -c ws://localhost:8998/v1/realtime
# Expect: {"type":"session.created","session":{...}}
```

**Validates:** C++ compiles, WebSocket handshake, JSON parsing, session lifecycle,
audio ring buffer, base64 encode/decode, graceful Triton error handling.

### Strategy B: Run Triton in CPU Mode

```bash
# Drop the GPU reservation and run Triton on CPU (very slow, for plumbing only)
docker run --rm -it \
  -v $(pwd)/model_repository:/models \
  -v $(pwd)/moshi:/app/moshi \
  -p 8000:8000 -p 8001:8001 \
  -e PERSONAPLEX_FP8=0 \
  nvcr.io/nvidia/tritonserver:25.02-py3 \
  bash -c "pip install -e /app/moshi && tritonserver \
    --model-repository=/models \
    --backend-config=python,shm-default-byte-size=134217728"
```

> **Warning:** CPU inference is 100-1000× slower than GPU. One 80ms audio frame takes
> 30-90 seconds. Requires ~64 GB of RAM. Useful only to confirm configs parse correctly
> and tensor shapes flow through the pipeline without error.

### Strategy C: Cloud GPU (Best for integration testing)

| GPU | VRAM | Max Sessions |
|-----|------|-------------|
| A100 80GB | 80 GB | 3-4 (set `PERSONAPLEX_FP8=0`) |
| H100 80GB | 80 GB | 4-5 (FP8 supported) |
| L40S 48GB | 48 GB | 2-3 (set `PERSONAPLEX_FP8=0`) |

Recommended providers: Lambda Labs, RunPod, Vast.ai, AWS p4d/p5.

### Strategy D: Unit Test Model Backends

```bash
pip install -e moshi/
python -c "
import torch
from moshi.models import loaders
device = torch.device('cpu')
mimi = loaders.get_mimi(loaders.get_mimi_weight_path(), device).eval()
mimi.streaming_forever(1)
codes = mimi.encode(torch.zeros(1, 1, 1920))
print('Encoder output:', codes.shape)   # expect [1, 8, 1]
decoded = mimi.decode(codes)
print('Decoder output:', decoded.shape) # expect [1, 1, 1920]
"
```

---

## 7. Deploying to DGX Spark

When your DGX Spark becomes available:

```bash
# SSH to Spark, pull the repo
ssh user@dgx-spark
git clone <repo-url> amarrmb-personaplex
cd amarrmb-personaplex

# Configure for Spark
cp .env.example .env
# Set: PERSONAPLEX_FP8=1, MAX_SESSIONS=6, HF_TOKEN=...

# Build and start (ARM64 native — ~15-30 min first time)
cd docker
docker compose build
docker compose up -d

# Verify
curl http://localhost:8000/v2/health/ready
docker compose logs gateway
```

---

## 8. API Reference

### Connection

```
WebSocket endpoint: ws[s]://host:8998/v1/realtime
```

### Client → Server Events

#### `session.update`
Configures the session. Must be sent before any audio. Server responds with
`session.updated` immediately, then `session.ready` after conditioning (~2-5s).
**Do not send audio until `session.ready` is received.**

```json
{
  "type": "session.update",
  "session": {
    "instructions": "You are a helpful assistant.",
    "voice_prompt_embedding": "<base64-encoded .pt file bytes>",
    "temperature": 0.8,
    "top_k": 250
  }
}
```

#### `input_audio_buffer.append`
Stream 24kHz PCM16 mono audio. Send any chunk size; the gateway frames it at 80ms.

```json
{
  "type": "input_audio_buffer.append",
  "audio": "<base64 PCM16 mono 24kHz little-endian>"
}
```

#### `input_audio_buffer.clear`
Discard buffered audio not yet consumed.

```json
{"type": "input_audio_buffer.clear"}
```

#### `response.cancel`
Cancel and disconnect.

```json
{"type": "response.cancel"}
```

### Server → Client Events

| Event | When |
|-------|------|
| `session.created` | On WebSocket connect |
| `session.updated` | After `session.update` (immediate) |
| `session.ready` | After system-prompt conditioning completes |
| `response.audio.delta` | Every 80ms — contains base64 PCM16 48kHz audio |
| `response.audio.done` | Session ended |
| `error` | On any error |

### Audio Formats

| Direction | Rate | Format |
|-----------|------|--------|
| Client → Server | 24,000 Hz | PCM16 mono little-endian, base64 |
| Server → Client | 48,000 Hz | PCM16 mono little-endian, base64 |

### Session Lifecycle

```
Client                    Gateway                   Triton
  │                          │                         │
  │──── WebSocket connect ──►│                         │
  │◄─── session.created ─────│                         │
  │                          │                         │
  │──── session.update ─────►│                         │
  │◄─── session.updated ─────│                         │
  │                          │──── Triton START ───────►│ (~2-5s)
  │                          │◄─── SESSION_READY=true ──│
  │◄─── session.ready ───────│                         │
  │                          │                         │
  │──── audio.append ───────►│                         │
  │                          │──── send_frame ─────────►│ (80ms)
  │                          │◄─── PCM 48kHz ───────────│
  │◄─── audio.delta ─────────│                         │
  │         (repeat)         │         (repeat)        │
  │                          │                         │
  │──── WS close ───────────►│                         │
  │                          │──── Triton END ─────────►│
  │◄─── audio.done ──────────│                         │
```

---

## 9. Operations & Monitoring

### Useful Commands

```bash
# Follow logs
docker compose -f docker/docker-compose.yaml logs -f

# Check Triton health
curl -s http://localhost:8000/v2/health/ready

# Check model status
curl -s http://localhost:8000/v2/models/personaplex_pipeline | jq

# Prometheus metrics
curl -s http://localhost:8002/metrics

# Stop
docker compose down

# Stop and wipe HuggingFace weight cache
docker compose down -v
```

### Key Prometheus Metrics

| Metric | Meaning |
|--------|---------|
| `nv_inference_request_duration_us` | Per-model inference latency |
| `nv_inference_queue_duration_us` | Queue wait time (high = need more instances) |
| `nv_gpu_utilization` | GPU busy % |
| `nv_gpu_memory_used_bytes` | VRAM usage |

---

## 10. Tuning & Performance

### Memory Budget per Session (DGX Spark, FP8 enabled)

| Component | VRAM |
|-----------|------|
| Mimi encoder (per instance) | ~100 MB |
| Mimi decoder (per instance) | ~100 MB |
| PersonaPlex LM FP8 (per instance) | ~10 GB |
| KV cache (per instance) | ~1.5 GB |
| LavaSR v2 (shared, one total) | ~100 MB |
| **Total per session** | **~11.8 GB** |

6 sessions × 11.8 GB = ~71 GB. DGX Spark has 128 GB unified memory — leaves ~47 GB headroom.

### Increasing Session Count

1. Edit `.env`: `MAX_SESSIONS=8`
2. Edit `instance_group.count: 8` in all three `config.pbtxt` files
3. `docker compose restart`

Start at 6 and increase one at a time while monitoring VRAM with `nvidia-smi`.

### Latency Optimisations

1. Export LavaSR v2 to TensorRT (see Section 5, Step 5)
2. Confirm `PERSONAPLEX_FP8=1` on Blackwell
3. Lower `LM_TOP_K` (250 → 100) for faster sampling

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Gateway reports "Triton start failed" | Triton not yet ready or Triton crashed | Check `docker compose logs triton` for model load errors |
| Triton fails to load a model | `config.pbtxt` parse error or missing Python dep | Look for `error` lines in Triton logs |
| "Maximum concurrent sessions reached" | All 6 slots occupied | Increase `MAX_SESSIONS` or wait for sessions to expire |
| Inference extremely slow | Running on CPU / FP8 off | Check `nvidia-smi` inside the Triton container |
| Audio garbled or wrong pitch | Sample rate mismatch | Confirm client sends 24kHz, plays back 48kHz |
| `session.ready` never arrives | Voice prompt corrupt or conditioning hung | Try without `voice_prompt_embedding` first; check Triton logs |
| Gateway segfault | Old binary with BUG-6 | Rebuild: `docker compose build gateway && docker compose up -d gateway` |
| `CUDA out of memory` on start | Too many instances for available VRAM | Reduce `MAX_SESSIONS` and `instance_group.count` |
