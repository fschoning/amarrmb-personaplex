# PersonaPlex v2 — Deployment Guide

## Architecture

```
Clients ──WSS/JSON──► C++ Gateway :8998 ──gRPC──► Triton :8001
                                                        │
                                       ┌────────────────┤ ensemble pipeline
                                       │ mimi_encoder   │ (×6 instances)
                                       │ personaplex_lm │ (×6 instances, FP8)
                                       │ mimi_decoder   │ (×6 instances)
                                       └► lavasr_v2     │ (×1, TensorRT)
```

## Quick Start (DGX Spark)

```bash
# 1. Clone and configure
git clone <this-repo>
cd amarrmb-personaplex
cp .env.example .env
# Edit .env — set HF_TOKEN if Moshi weights are gated

# 2. (Optional) Pre-export LavaSR v2 to TensorRT
docker run --gpus all --rm \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:25.02-py3 \
  python /app/scripts/export_lavasr.py \
    --out-dir /models/lavasr_v2/1
# Then update model_repository/lavasr_v2/config.pbtxt: backend "python" → "tensorrt"

# 3. Launch
cd docker
docker compose up --build -d

# 4. Check health
curl http://localhost:8000/v2/health/ready   # Triton
# Gateway logs: "Listening on ws://0.0.0.0:8998/v1/realtime"
```

## API — OpenAI Realtime API (subset)

WebSocket endpoint: `ws[s]://host:8998/v1/realtime`

### Session setup
```json
// → session.update
{
  "type": "session.update",
  "session": {
    "instructions": "You are a helpful assistant.",
    "voice_prompt_embedding": "<base64-encoded .pt bytes>",
    "input_audio_format": "pcm16",
    "output_audio_format": "pcm16",
    "temperature": 0.8,
    "top_k": 250
  }
}

// ← session.updated  (immediate)
// ← session.ready    (after ~2-5s system-prompt conditioning)
```

### Audio streaming (after session.ready)
```json
// → input_audio_buffer.append  (every 20ms or 80ms)
{
  "type": "input_audio_buffer.append",
  "audio": "<base64 PCM16 24kHz mono LE>"
}

// ← response.audio.delta  (every 80ms)
{
  "type": "response.audio.delta",
  "response_id": "resp_...",
  "item_id": "item_...",
  "delta": "<base64 PCM16 48kHz mono LE>"
}
```

### Uploading a voice prompt

Voice prompts are `.pt` files from the original PersonaPlex voice library.
```python
import base64, pathlib, json, websockets, asyncio

async def connect():
    async with websockets.connect("ws://host:8998/v1/realtime") as ws:
        # Read session.created
        await ws.recv()
        
        # Upload voice prompt
        pt_bytes = pathlib.Path("voices/expresso.pt").read_bytes()
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "voice_prompt_embedding": base64.b64encode(pt_bytes).decode(),
                "instructions": "You are a helpful assistant.",
                "temperature": 0.8
            }
        }))
        
        # Wait for session.ready
        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "session.ready":
                break
        
        # Start streaming PCM16 24kHz audio chunks...
```

## Tuning concurrent sessions

`MAX_SESSIONS` in `.env` controls both:
1. The gateway session cap (rejects connections beyond this)
2. Should match `instance_group.count` in `mimi_encoder/config.pbtxt`,
   `personaplex_lm/config.pbtxt`, and `mimi_decoder/config.pbtxt`

Memory budget per session (DGX Spark, 128GB unified):
| Component | VRAM per instance |
|-----------|------------------|
| Mimi encoder + decoder | ~200 MB |
| LM (FP8) | ~10 GB |
| KV cache | ~1.5 GB |
| LavaSR v2 (shared) | ~100 MB total |
| **Total per session** | **~11.8 GB** |

Start at 6 (default), profile, and raise if memory headroom allows.

## Phase 2 — Full Python Elimination (optional)

See `design_document.md §8`. Converts Mimi and the LM to TensorRT.
Run `scripts/export_mimi.py` (scaffold provided) and `scripts/export_lavasr.py`.
Expected latency improvement: ~2-3%. Estimated effort: 13-20 additional person-weeks.
