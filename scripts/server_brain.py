#!/usr/bin/env python3
"""
scripts/server_brain.py — FastAPI HTTP server for SmolLM3-3B TRT-LLM Brain

Runs inside the spark-single-gpu-dev container.
Loads the NVFP4 TRT-LLM engine and serves HTTP requests.

Endpoint:
    POST /generate
    Body: {"prompt": "...", "max_tokens": 64}
    Response: {"response": "...", "elapsed_s": 0.3}

    GET /health
    Response: {"status": "ready"}
"""

import os
import sys
import time
import re
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [brain] %(message)s")
log = logging.getLogger("brain")

# ── Config ───────────────────────────────────────────────────────────────────
ENGINE_DIR = os.environ.get("BRAIN_ENGINE", "/mnt/models/smollm3-3b-nvfp4-engine")
MODEL_DIR  = os.environ.get("BRAIN_MODEL",  "/mnt/models/smollm3-3b")
PORT       = int(os.environ.get("BRAIN_PORT", "8015"))

# ── Load model ───────────────────────────────────────────────────────────────
engine = None
tokenizer = None
backend = None


def load_model():
    global engine, tokenizer, backend

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # Try TRT-LLM C++ engine first
    if os.path.isdir(ENGINE_DIR) and os.path.isfile(os.path.join(ENGINE_DIR, "rank0.engine")):
        try:
            from tensorrt_llm._tensorrt_engine import LLM as TrtLLM
            engine = TrtLLM(
                model=ENGINE_DIR,
                tokenizer=MODEL_DIR,
                kv_cache_config={"free_gpu_memory_fraction": 0.05},
            )
            backend = "trtllm-cpp"
            log.info(f"Loaded TRT-LLM C++ engine from {ENGINE_DIR}")
            return
        except Exception as e:
            log.warning(f"TRT-LLM C++ engine failed: {e}")

        # Fallback: ModelRunner
        try:
            from tensorrt_llm.runtime import ModelRunner
            engine = ModelRunner.from_dir(engine_dir=ENGINE_DIR, rank=0)
            backend = "trtllm-runner"
            log.info(f"Loaded TRT-LLM ModelRunner from {ENGINE_DIR}")
            return
        except Exception as e:
            log.warning(f"TRT-LLM ModelRunner failed: {e}")

    # Fallback: HF transformers
    try:
        import torch
        from transformers import AutoModelForCausalLM
        engine = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=torch.bfloat16,
            device_map="cuda", trust_remote_code=True,
        )
        engine.eval()
        backend = "transformers"
        log.info(f"Loaded via transformers (bf16) from {MODEL_DIR}")
    except Exception as e:
        log.error(f"All backends failed: {e}")
        raise


def format_prompt(prompt: str) -> str:
    """Format using tokenizer's chat template."""
    if "<|im_start|>" in prompt or "[INST]" in prompt or "<|system|>" in prompt:
        return prompt
    messages = [
        {"role": "system", "content": "You are a concise AI assistant. Respond directly."},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return (
            "<|im_start|>system\nYou are a concise AI assistant. Respond directly.\n<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )


def strip_thinking(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
    return text.strip()


def generate(prompt: str, max_tokens: int = 64) -> str:
    formatted = format_prompt(prompt)

    if backend == "trtllm-cpp":
        from tensorrt_llm import SamplingParams as TrtSamplingParams
        eos_id = tokenizer.eos_token_id or 2
        params = TrtSamplingParams(
            max_tokens=max_tokens, temperature=0.7, top_p=0.9,
            end_id=eos_id, pad_id=eos_id,
        )
        outputs = engine.generate([formatted], sampling_params=params)
        if hasattr(outputs[0], 'text'):
            raw = outputs[0].text
        elif hasattr(outputs[0], 'outputs'):
            raw = outputs[0].outputs[0].text
        else:
            raw = str(outputs[0])
        return strip_thinking(raw.strip())

    elif backend == "trtllm-runner":
        import torch
        input_ids = tokenizer.encode(formatted, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = engine.generate(
                batch_input_ids=[input_ids[0]],
                max_new_tokens=max_tokens,
                temperature=0.7, top_p=0.9,
            )
        raw = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        return strip_thinking(raw.strip())

    elif backend == "transformers":
        import torch
        inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = engine.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=0.7, top_p=0.9, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
        return strip_thinking(raw.strip())


# ── FastAPI server ───────────────────────────────────────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="PersonaPlex Brain — SmolLM3-3B")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 64


class GenerateResponse(BaseModel):
    response: str
    elapsed_s: float
    backend: str


@app.on_event("startup")
def startup():
    log.info("Loading model...")
    load_model()
    # Warmup
    log.info("Warming up...")
    t0 = time.monotonic()
    result = generate("Say hello.", 8)
    log.info(f"Warmup done in {time.monotonic() - t0:.2f}s: '{result[:60]}'")
    log.info(f"Brain ready (backend={backend}, port={PORT})")


@app.get("/health")
def health():
    return {"status": "ready", "backend": backend}


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    t0 = time.monotonic()
    result = generate(req.prompt, req.max_tokens)
    elapsed = time.monotonic() - t0
    n_words = len(result.split())
    log.info(f"Generated {n_words} words in {elapsed:.2f}s ({n_words/max(elapsed,0.01):.0f} w/s)")
    return GenerateResponse(response=result, elapsed_s=round(elapsed, 3), backend=backend)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
