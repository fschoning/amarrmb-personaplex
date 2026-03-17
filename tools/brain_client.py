#!/usr/bin/env python3
"""
PersonaPlex v3 — Workstation Brain Client
==========================================
Connects to the PersonaPlex gateway on the Spark.
Receives: AI audio (plays to speakers) + transcript text (feeds brain)
Sends:    Human mic audio + node commands (switch, prime, etc.)

The Brain runs here on the workstation using Google Gemini with context
caching — the growing transcript is cached so we only pay for new tokens.

Usage:
  pip install websockets sounddevice numpy google-generativeai

  # Interactive conversation with brain:
  python brain_client.py --host 192.168.2.117 --port 8998 \\
      --persona "You are Jane, a curious AI who loves science." \\
      --filler "Hold on, let me check that for you." \\
      --voice NATF0

  # Set API key via env:
  export GOOGLE_API_KEY=AIza...
  python brain_client.py --host 192.168.2.117

  # Without brain (audio test only):
  python brain_client.py --host 192.168.2.117 --no-brain
"""

import argparse
import asyncio
import base64
import datetime
import json
import os
import queue
import sys
import threading
import time

try:
    import websockets
except ImportError:
    print("ERROR: pip install websockets"); sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: pip install numpy"); sys.exit(1)

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None
    print("WARNING: sounddevice not available — mic/speaker mode disabled")

try:
    import google.generativeai as genai
    from google.generativeai import caching as genai_caching
    _GENAI = True
except ImportError:
    _GENAI = False
    print("WARNING: pip install google-generativeai — brain will be disabled")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_RATE   = 24000
OUTPUT_RATE  = 48000
FRAME_MS     = 80
FRAME_SAMPLES_IN  = INPUT_RATE  * FRAME_MS // 1000   # 1920
FRAME_SAMPLES_OUT = OUTPUT_RATE * FRAME_MS // 1000   # 3840

GEMINI_MODEL = "gemini-2.0-flash-001"

# Context caching:
# - Gemini requires ≥1024 tokens to be cached
# - We refresh cache every CACHE_REFRESH_CHARS new chars (~1 token ≈ 4 chars)
# - Min cache TTL is 1 hour; we renew each refresh
CACHE_MIN_CHARS    = 4096   # ~1024 tokens — minimum to enable caching
CACHE_REFRESH_CHARS = 2000  # Refresh cache every ~500 new tokens
CACHE_TTL_HOURS    = 1

# Brain query throttle
BRAIN_MIN_CHARS  = 100    # At least 100 chars of transcript before first query
BRAIN_INTERVAL_S = 10.0   # Don't query faster than this

VOICE_NAMES = [
    "NATF0","NATF1","NATF2","NATF3",
    "NATM0","NATM1","NATM2","NATM3",
    "VARF0","VARF1","VARF2","VARF3","VARF4",
    "VARM0","VARM1","VARM2","VARM3","VARM4",
]
VOICE_SENTINEL = -999

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def pcm16_to_b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")

def b64_to_pcm16(b64: str) -> bytes:
    return base64.b64decode(b64)

def pcm16_to_float32(data: bytes):
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

def encode_voice_tokens(voice_name: str) -> list:
    return [VOICE_SENTINEL] + [ord(c) for c in voice_name]

# ---------------------------------------------------------------------------
# Transcript accumulator
# ---------------------------------------------------------------------------

class Transcript:
    """Thread-safe rolling transcript buffer with delta tracking."""

    def __init__(self):
        self._lock  = threading.Lock()
        self._text  = ""
        # Track how many chars were in the last cache upload
        self._cached_len = 0

    def append(self, delta: str):
        with self._lock:
            self._text += delta
            if len(self._text) > 100_000:
                # Trim oldest — but keep _cached_len valid
                trim = len(self._text) - 80_000
                self._text = self._text[trim:]
                self._cached_len = max(0, self._cached_len - trim)

    def get(self) -> str:
        with self._lock:
            return self._text

    def new_chars_since_cache(self) -> int:
        with self._lock:
            return len(self._text) - self._cached_len

    def mark_cached(self):
        with self._lock:
            self._cached_len = len(self._text)

    def clear(self):
        with self._lock:
            self._text = ""
            self._cached_len = 0

    def __len__(self):
        with self._lock:
            return len(self._text)

# ---------------------------------------------------------------------------
# Brain — Google Gemini with context caching
# ---------------------------------------------------------------------------

class Brain:
    """
    Calls Gemini Flash using Google's native API with context caching.

    Strategy:
      - The SYSTEM INSTRUCTION (persona + format rules) is always cached.
      - The growing TRANSCRIPT is uploaded as cached content and refreshed
        every CACHE_REFRESH_CHARS new characters.
      - Each brain query only sends the small "delta" prompt (new content
        since last cache), keeping per-query token costs minimal.
    
    Token cost model:
      Full price:  new transcript delta + query instruction (~50-100 tokens)
      Cache price: cached system + transcript (~25% normal rate)
    """

    def __init__(self, api_key: str, model: str = GEMINI_MODEL, persona: str = ""):
        genai.configure(api_key=api_key)
        self.model_name = f"models/{model}"
        self.persona    = persona
        self._cache     = None      # CachedContent object
        self._cache_transcript_len = 0  # chars in cache at last refresh
        self._lock      = threading.Lock()

        self._system_instruction = self._build_system_instruction()
        print(f"  [brain] Using model: {model}")
        print(f"  [brain] Context caching: enabled (min {CACHE_MIN_CHARS} chars = ~1K tokens)")

    def _build_system_instruction(self) -> str:
        return f"""You are the brain behind a voice AI agent.
AI Persona: {self.persona}

Your job: analyse the transcript of what the AI has been saying, then write
a BOOT_PAYLOAD for the next AI node so the conversation continues seamlessly.

Rules:
- Respond ONLY with the structure below, no markdown, no preamble.
- Be concise — each field max 2 sentences.

[SUMMARY] <what the conversation has been about>
[CONTEXT] <key facts, names, topics the AI should know>
[LAST_TOPIC] <the most recent topic or question being addressed>
[EMOTION] <tone to match: e.g. warm, curious, professional, excited>
[PERSONA] {self.persona}"""

    def _create_or_refresh_cache(self, transcript: str):
        """Upload the full transcript as cached content. Called when cache is
        stale or doesn't exist. Thread must hold self._lock."""
        # Build the cached content — transcript as a 'user' turn
        contents = [
            genai.protos.Content(
                role="user",
                parts=[genai.protos.Part(text=f"=== FULL TRANSCRIPT ===\n{transcript}\n=== END ===")]
            )
        ]

        # Delete old cache if exists
        if self._cache:
            try:
                self._cache.delete()
            except Exception:
                pass
            self._cache = None

        try:
            self._cache = genai_caching.CachedContent.create(
                model=self.model_name,
                display_name="personaplex_transcript",
                system_instruction=self._system_instruction,
                contents=contents,
                ttl=datetime.timedelta(hours=CACHE_TTL_HOURS),
            )
            self._cache_transcript_len = len(transcript)
            print(f"\n  [brain] Cache created/refreshed "
                  f"({len(transcript)} chars ≈ {len(transcript)//4} tokens)")
        except Exception as e:
            # Caching failed (e.g. transcript too short) — fall back to no-cache
            print(f"\n  [brain] Cache create failed ({e}) — using direct call")
            self._cache = None

    def query(self, transcript: str, delta_chars: int) -> str:
        """
        Query Gemini. Uses cached context when transcript is large enough.
        delta_chars: how many chars are new since the last cache refresh.
        """
        with self._lock:
            use_cache = (
                len(transcript) >= CACHE_MIN_CHARS
                and delta_chars >= CACHE_REFRESH_CHARS
            )

            if use_cache:
                # Refresh cache with full transcript
                self._create_or_refresh_cache(transcript)

            if self._cache:
                # Query using cached context — only send the instruction
                model = genai.GenerativeModel.from_cached_content(self._cache)
                query_text = (
                    "Based on the cached transcript above, write the BOOT_PAYLOAD "
                    "for the next AI node. Follow the format exactly."
                )
            else:
                # No cache (transcript too short or caching unavailable)
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=self._system_instruction,
                )
                # Include full transcript in the prompt
                recent = transcript[-4000:] if len(transcript) > 4000 else transcript
                query_text = (
                    f"=== TRANSCRIPT ===\n{recent}\n=== END ===\n\n"
                    "Write the BOOT_PAYLOAD for the next AI node."
                )

        response = model.generate_content(
            query_text,
            generation_config=genai.GenerationConfig(
                max_output_tokens=300,
                temperature=0.3,
            ),
        )
        return response.text.strip()

    def cleanup(self):
        """Delete cached content on shutdown."""
        with self._lock:
            if self._cache:
                try:
                    self._cache.delete()
                    print("  [brain] Cache deleted.")
                except Exception:
                    pass

# ---------------------------------------------------------------------------
# PersonaPlex v3 Client
# ---------------------------------------------------------------------------

class PersonaPlexClient:

    def __init__(self, args):
        self.args       = args
        self.ws         = None
        self.loop       = None
        self.transcript = Transcript()
        self.node_state = {"active": True, "standby": False,
                           "filler": False, "state": "hot_only"}

        self._audio_out_q: queue.Queue = queue.Queue(maxsize=200)

        # Brain
        self.brain = None
        if not args.no_brain and _GENAI and args.google_api_key:
            self.brain = Brain(
                api_key=args.google_api_key,
                model=args.model,
                persona=args.persona,
            )
        elif not args.no_brain:
            if not _GENAI:
                print("  Brain: disabled — pip install google-generativeai")
            else:
                print("  Brain: disabled — set GOOGLE_API_KEY or --google-api-key")
        else:
            print("  Brain: disabled (--no-brain)")

        self._brain_lock      = threading.Lock()
        self._brain_in_flight = False
        self._last_brain_t    = 0.0

    # ── WebSocket helpers ─────────────────────────────────────────────────

    async def _send(self, msg: dict):
        if self.ws:
            await self.ws.send(json.dumps(msg))

    async def send_node_switch(self, to: str):
        print(f"\n  → node.switch to={to}")
        await self._send({"type": "node.switch", "to": to})

    async def send_node_prime(self, prompt: str):
        print(f"\n  → node.prime: {prompt[:80]}...")
        await self._send({"type": "node.prime", "target": "standby", "prompt": prompt})

    async def send_node_stop(self, target: str = "filler"):
        print(f"\n  → node.stop target={target}")
        await self._send({"type": "node.stop", "target": target})

    # ── Brain worker ──────────────────────────────────────────────────────

    def _brain_worker(self, transcript_snapshot: str, delta_chars: int):
        """Run in thread. Queries Gemini (with caching) and primes standby."""
        try:
            t0 = time.monotonic()
            response = self.brain.query(transcript_snapshot, delta_chars)
            elapsed  = time.monotonic() - t0

            # Mark transcript as cached up to this point
            self.transcript.mark_cached()

            cached_flag = "📦 cached" if len(transcript_snapshot) >= CACHE_MIN_CHARS else "📝 direct"
            print(f"\n  [brain] {elapsed:.1f}s {cached_flag} → {response[:100]}...")

            if self.loop and response:
                asyncio.run_coroutine_threadsafe(
                    self.send_node_prime(response), self.loop
                )

        except Exception as e:
            print(f"\n  [brain] ERROR: {e}")
        finally:
            with self._brain_lock:
                self._brain_in_flight = False

    def _maybe_query_brain(self):
        if not self.brain:
            return
        now = time.monotonic()
        if now - self._last_brain_t < BRAIN_INTERVAL_S:
            return
        if len(self.transcript) < BRAIN_MIN_CHARS:
            return
        with self._brain_lock:
            if self._brain_in_flight:
                return
            self._brain_in_flight = True

        self._last_brain_t = now
        snapshot    = self.transcript.get()
        delta_chars = self.transcript.new_chars_since_cache()

        threading.Thread(
            target=self._brain_worker,
            args=(snapshot, delta_chars),
            daemon=True,
        ).start()

    # ── Message handler ───────────────────────────────────────────────────

    async def _on_message(self, raw: str):
        try:
            ev = json.loads(raw)
        except Exception:
            return

        t = ev.get("type", "")

        if t == "session.created":
            sid = ev.get("session", {}).get("id", ev.get("id", "?"))
            print(f"  session.created  id={sid}")

        elif t == "session.updated":
            print("  session.updated")

        elif t == "session.ready":
            print("  session.ready ✓  —  PP nodes loaded, audio flowing...")

        elif t == "response.audio.delta":
            b64 = ev.get("delta", "")
            if b64:
                pcm = pcm16_to_float32(b64_to_pcm16(b64))
                try:
                    self._audio_out_q.put_nowait(pcm)
                except queue.Full:
                    pass

        elif t == "transcript.delta":
            text = ev.get("text", "")
            if text:
                self.transcript.append(text)
                print(f"\r  AI: ...{self.transcript.get()[-70:]}", end="", flush=True)
                self._maybe_query_brain()

        elif t == "node.status":
            self.node_state = {k: ev.get(k) for k in
                               ("active", "standby", "filler", "state")}
            s = self.node_state
            print(f"\n  [node.status] active={s['active']} standby={s['standby']} "
                  f"filler={s['filler']} state={s['state']}")

        elif t == "node.standby_ready":
            print("\n  [node.standby_ready] — issuing node.switch to standby...")
            await self.send_node_switch("standby")

        elif t in ("error", "session_error"):
            msg = ev.get("error", {}).get("message", ev.get("message", "?"))
            print(f"\n  [ERROR] {msg}")

    # ── Mic sender ────────────────────────────────────────────────────────

    async def _mic_sender(self):
        if not sd:
            await asyncio.sleep(99999)
            return

        mic_q: queue.Queue = queue.Queue()

        def callback(indata, frames, time_cb, status):
            mic_q.put_nowait(bytes(indata))

        with sd.RawInputStream(samplerate=INPUT_RATE, channels=1, dtype="int16",
                                blocksize=FRAME_SAMPLES_IN, callback=callback):
            while True:
                try:
                    raw = mic_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                await self._send({
                    "type":  "input_audio_buffer.append",
                    "audio": pcm16_to_b64(raw),
                })

    # ── Speaker output ────────────────────────────────────────────────────

    def _speaker_thread(self):
        if not sd:
            return
        stream = sd.OutputStream(samplerate=OUTPUT_RATE, channels=1, dtype="float32")
        stream.start()
        try:
            while True:
                try:
                    pcm = self._audio_out_q.get(timeout=0.5)
                    stream.write(pcm.reshape(-1, 1))
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"\n[speaker] {e}")
        finally:
            stream.stop()
            stream.close()

    # ── Keyboard commands ─────────────────────────────────────────────────

    async def _command_loop(self):
        print("""
  Commands:
    s        — switch to standby (if primed)
    f        — switch to filler (immediate)
    r        — reset filler
    p <text> — prime standby with custom prompt
    b        — trigger brain query now
    t        — print full transcript
    n        — print node status
    q        — quit
""")
        loop = asyncio.get_event_loop()
        while True:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
            except Exception:
                break
            line = line.strip()
            if not line:
                continue
            if line == "q":
                if self.ws:
                    await self.ws.close()
                break
            elif line == "s":
                await self.send_node_switch("standby")
            elif line == "f":
                await self.send_node_switch("filler")
            elif line == "r":
                await self.send_node_stop("filler")
            elif line.startswith("p "):
                await self.send_node_prime(line[2:].strip())
            elif line == "b":
                # Force brain query
                self._last_brain_t = 0.0
                self._maybe_query_brain()
                print("  [brain] Query triggered.")
            elif line == "t":
                print(f"\n--- Transcript ---\n{self.transcript.get()}\n---")
            elif line == "n":
                print(f"\n  Node state: {self.node_state}")
            else:
                print(f"  Unknown command: {line!r}")

    # ── Session setup ─────────────────────────────────────────────────────

    async def _send_session_update(self):
        session: dict = {
            "instructions":  self.args.persona,
            "filler_prompt": self.args.filler,
        }
        v = self.args.voice.upper()
        if v in VOICE_NAMES:
            session["text_prompt_tokens"] = encode_voice_tokens(v)

        await self._send({"type": "session.update", "session": session})
        print("  Sent session.update, waiting for session.ready...")

    # ── Main ──────────────────────────────────────────────────────────────

    async def run(self):
        url = f"ws://{self.args.host}:{self.args.port}/v1/realtime"
        print(f"Connecting to {url} ...")

        threading.Thread(target=self._speaker_thread, daemon=True).start()
        self.loop = asyncio.get_event_loop()

        try:
            async with websockets.connect(url, max_size=32*1024*1024) as ws:
                self.ws = ws
                print("Connected!")
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                await self._on_message(raw)
                await self._send_session_update()
                await asyncio.gather(
                    self._receive_loop(),
                    self._mic_sender(),
                    self._command_loop(),
                )
        except websockets.exceptions.ConnectionRefusedError:
            print(f"\nERROR: Connection refused at {url}")
        except asyncio.TimeoutError:
            print("\nERROR: Timeout waiting for server")
        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            if self.brain:
                self.brain.cleanup()

    async def _receive_loop(self):
        async for msg in self.ws:
            await self._on_message(msg)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pick_voice() -> str:
    groups = [
        ("Natural Female", ["NATF0","NATF1","NATF2","NATF3"]),
        ("Natural Male",   ["NATM0","NATM1","NATM2","NATM3"]),
        ("Varied Female",  ["VARF0","VARF1","VARF2","VARF3","VARF4"]),
        ("Varied Male",    ["VARM0","VARM1","VARM2","VARM3","VARM4"]),
    ]
    print("\n  Available voices:")
    idx, flat = 1, []
    for name, voices in groups:
        print(f"    ── {name} ──")
        for v in voices:
            print(f"    {idx:2d}. {v}")
            flat.append(v)
            idx += 1
    print("     0. (no voice prompt — use default)\n")
    while True:
        try:
            sel = int(input("  Select voice [1-18, 0=default]: ").strip())
            if sel == 0: return ""
            if 1 <= sel <= len(flat):
                print(f"  Selected: {flat[sel-1]}")
                return flat[sel-1]
        except (ValueError, EOFError):
            pass


def main():
    p = argparse.ArgumentParser(description="PersonaPlex v3 Brain Client")
    p.add_argument("--host",  default="192.168.2.117")
    p.add_argument("--port",  type=int, default=8998)
    p.add_argument("--persona", default="You are a helpful, friendly AI assistant.")
    p.add_argument("--filler",  default="Hold on, I'm working on that for you. Just a moment.")
    p.add_argument("--voice",   default="", help="Voice e.g. NATF0, or blank for interactive")
    p.add_argument("--google-api-key",
                   default=os.environ.get("GOOGLE_API_KEY", ""),
                   help="Google AI API key (or set GOOGLE_API_KEY env var)")
    p.add_argument("--model", default=GEMINI_MODEL)
    p.add_argument("--no-brain", action="store_true")
    args = p.parse_args()

    print(f"""
PersonaPlex v3 Brain Client
  Server  : {args.host}:{args.port}
  Persona : {args.persona[:60]}
  Filler  : {args.filler[:60]}
  Brain   : {'disabled' if args.no_brain else f'Gemini {args.model} + context caching'}
""")

    if not args.voice:
        args.voice = pick_voice()

    try:
        asyncio.run(PersonaPlexClient(args).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
