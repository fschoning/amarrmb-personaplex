#!/usr/bin/env python3
"""
scripts/server_brain.py — FastAPI proxy to OpenRouter for PersonaPlex Brain

Thin HTTP proxy: receives requests from the C++ gateway BrainClient,
forwards them to OpenRouter (Gemini 2.0 Flash), returns the response.

Same API as before:
    POST /generate  {"prompt": "...", "max_tokens": 200}
    GET  /health    → {"status": "ready"}

Environment:
    OPENROUTER_API_KEY  — required
    OPENROUTER_MODEL    — default: google/gemini-2.0-flash-001
    BRAIN_PORT          — default: 8015
    BRAIN_MAX_TOKENS    — default: 200
    BRAIN_TEMPERATURE   — default: 0.7
"""

import os
import time
import logging
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [brain] %(message)s")
log = logging.getLogger("brain")

# ── Config ───────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
PORT               = int(os.environ.get("BRAIN_PORT", "8015"))
DEFAULT_MAX_TOKENS = int(os.environ.get("BRAIN_MAX_TOKENS", "200"))
TEMPERATURE        = float(os.environ.get("BRAIN_TEMPERATURE", "0.7"))

# Persistent HTTP client for connection pooling
_client = httpx.Client(timeout=30.0)


# ── OpenRouter API call ──────────────────────────────────────────────────────
def query_openrouter(prompt: str, max_tokens: int = None) -> str:
    """Send a prompt to OpenRouter and return the response text."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")

    max_tokens = max_tokens or DEFAULT_MAX_TOKENS

    # Build messages — if prompt already contains role markers, wrap as-is
    # Otherwise create a simple user message
    messages = [
        {
            "role": "system",
            "content": (
                "You are a concise AI brain that analyzes conversations and "
                "provides routing decisions. Respond directly and concisely."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://personaplex.ai",
        "X-Title": "PersonaPlex Brain",
    }

    response = _client.post(OPENROUTER_URL, json=payload, headers=headers)
    response.raise_for_status()

    data = response.json()

    # Extract response text
    choices = data.get("choices", [])
    if not choices:
        log.warning(f"OpenRouter returned no choices: {data}")
        return ""

    message = choices[0].get("message", {})
    content = message.get("content", "").strip()

    # Log usage
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    log.info(
        f"OpenRouter: model={OPENROUTER_MODEL}, "
        f"prompt={prompt_tokens} toks, completion={completion_tokens} toks"
    )

    return content


# ── FastAPI server ───────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="PersonaPlex Brain — OpenRouter Proxy")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = None


class GenerateResponse(BaseModel):
    response: str
    elapsed_s: float
    backend: str


@app.on_event("startup")
def startup():
    if not OPENROUTER_API_KEY:
        log.error("OPENROUTER_API_KEY not set! Brain will not work.")
        return

    log.info(f"Brain proxy starting...")
    log.info(f"  Model:    {OPENROUTER_MODEL}")
    log.info(f"  Port:     {PORT}")
    log.info(f"  Max toks: {DEFAULT_MAX_TOKENS}")

    # Quick test call to verify API key works
    log.info("Testing OpenRouter connection...")
    try:
        t0 = time.monotonic()
        result = query_openrouter("Say hello in one sentence.", 20)
        elapsed = time.monotonic() - t0
        log.info(f"OpenRouter test OK in {elapsed:.2f}s: '{result[:80]}'")
        log.info("Brain ready.")
    except Exception as e:
        log.error(f"OpenRouter test FAILED: {e}")
        log.error("Check OPENROUTER_API_KEY. Brain will retry on first real request.")


@app.get("/health")
def health():
    return {
        "status": "ready" if OPENROUTER_API_KEY else "no_api_key",
        "model": OPENROUTER_MODEL,
        "backend": "openrouter",
    }


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY not set")

    t0 = time.monotonic()
    try:
        result = query_openrouter(req.prompt, req.max_tokens)
    except httpx.HTTPStatusError as e:
        log.error(f"OpenRouter HTTP error: {e.response.status_code} — {e.response.text[:200]}")
        raise HTTPException(status_code=502, detail=f"OpenRouter error: {e.response.status_code}")
    except Exception as e:
        log.error(f"OpenRouter error: {e}")
        raise HTTPException(status_code=502, detail=str(e))

    elapsed = time.monotonic() - t0
    n_words = len(result.split())
    log.info(
        f"Generated {n_words} words in {elapsed:.2f}s "
        f"({n_words / max(elapsed, 0.01):.0f} w/s)"
    )

    return GenerateResponse(
        response=result,
        elapsed_s=round(elapsed, 3),
        backend=f"openrouter/{OPENROUTER_MODEL}",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
