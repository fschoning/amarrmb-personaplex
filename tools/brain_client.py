#!/usr/bin/env python3
"""
PersonaPlex v3 — Workstation Brain Client
==========================================
Connects to the PersonaPlex gateway on the Spark.
Receives: AI audio (plays to speakers) + transcript text (feeds brain)
Sends:    Human mic audio + node commands (switch, prime, etc.)

The Brain runs here on the workstation and calls Gemini Flash via OpenRouter.

Usage:
  pip install websockets sounddevice numpy httpx

  # Interactive conversation with brain:
  python brain_client.py --host 192.168.2.117 --port 8998 \\
      --persona "You are Jane, a curious AI who loves science." \\
      --filler "Hold on, let me check that for you." \\
      --voice NATF0 \\
      --openrouter-key sk-or-v1-...

  # Without brain (audio test only):
  python brain_client.py --host 192.168.2.117 --port 8998 --no-brain
"""

import argparse
import asyncio
import base64
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime

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
    import httpx
    _HTTPX = True
except ImportError:
    _HTTPX = False
    print("WARNING: pip install httpx — brain will be disabled")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_RATE   = 24000   # Server expects 24kHz PCM16
OUTPUT_RATE  = 48000   # Server outputs 48kHz
FRAME_MS     = 80
FRAME_SAMPLES_IN  = INPUT_RATE  * FRAME_MS // 1000   # 1920
FRAME_SAMPLES_OUT = OUTPUT_RATE * FRAME_MS // 1000   # 3840

OPENROUTER_URL   = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemini-2.0-flash-001"

# Minimum transcript chars before querying brain
BRAIN_MIN_CHARS  = 50
# Query brain every N seconds (not every frame)
BRAIN_INTERVAL_S = 8.0

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

def float32_to_pcm16(arr) -> bytes:
    return np.clip(arr, -1., 1.).multiply(32767).astype(np.int16).tobytes() \
        if False else (np.clip(arr, -1., 1.) * 32767).astype(np.int16).tobytes()

def pcm16_to_float32(data: bytes):
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

def encode_voice_tokens(voice_name: str) -> list:
    """Encode voice name as sentinel + ASCII ints for PP prompt injection."""
    return [VOICE_SENTINEL] + [ord(c) for c in voice_name]

# ---------------------------------------------------------------------------
# Brain (Gemini Flash via OpenRouter)
# ---------------------------------------------------------------------------

class Brain:
    """Calls Gemini Flash via OpenRouter to analyse the transcript and decide node actions."""

    def __init__(self, api_key: str, model: str = OPENROUTER_MODEL):
        self.api_key = api_key
        self.model   = model
        self._client = httpx.Client(timeout=30.0)

    def query(self, system: str, user: str, max_tokens: int = 300) -> str:
        """Synchronous query — run in a thread."""
        resp = self._client.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/personaplex",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "max_tokens":   max_tokens,
                "temperature":  0.3,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def close(self):
        self._client.close()


# ---------------------------------------------------------------------------
# Transcript accumulator
# ---------------------------------------------------------------------------

class Transcript:
    def __init__(self):
        self._lock = threading.Lock()
        self._text = ""

    def append(self, delta: str):
        with self._lock:
            self._text += delta
            # Keep last 50K chars
            if len(self._text) > 50000:
                self._text = self._text[-40000:]

    def get(self) -> str:
        with self._lock:
            return self._text

    def clear(self):
        with self._lock:
            self._text = ""

    def __len__(self):
        with self._lock:
            return len(self._text)


# ---------------------------------------------------------------------------
# PersonaPlex v3 Brain Client
# ---------------------------------------------------------------------------

class PersonaPlexClient:

    def __init__(self, args):
        self.args       = args
        self.ws         = None
        self.loop       = None
        self.transcript = Transcript()
        self.node_state = {"active": True, "standby": False, "filler": False, "state": "hot_only"}

        # Output audio queue (filled by WebSocket receiver, drained by sounddevice)
        self._audio_out_q: queue.Queue = queue.Queue(maxsize=100)

        # Brain
        self.brain = None
        if args.openrouter_key and _HTTPX and not args.no_brain:
            self.brain = Brain(args.openrouter_key, args.model)
            print(f"  Brain: Gemini Flash ({args.model})")
        else:
            print("  Brain: disabled (--no-brain or missing --openrouter-key)")

        self._brain_lock      = threading.Lock()
        self._brain_in_flight = False
        self._last_brain_t    = 0.0
        self._standby_primed  = False

    # ── WebSocket send helpers ─────────────────────────────────────────────

    async def _send(self, msg: dict):
        if self.ws:
            await self.ws.send(json.dumps(msg))

    async def send_node_switch(self, to: str):
        """Switch to 'standby' or 'filler'."""
        print(f"  → node.switch to={to}")
        await self._send({"type": "node.switch", "to": to})

    async def send_node_prime(self, prompt: str, voice: str = ""):
        """Prime the standby node with new context."""
        print(f"  → node.prime standby: {prompt[:60]}...")
        msg = {"type": "node.prime", "target": "standby", "prompt": prompt}
        if voice:
            msg["voice"] = voice
        await self._send(msg)
        self._standby_primed = False  # will be set when node.standby_ready arrives

    async def send_node_stop(self, target: str = "filler"):
        print(f"  → node.stop target={target}")
        await self._send({"type": "node.stop", "target": target})

    # ── Brain decision loop ────────────────────────────────────────────────

    def _brain_worker(self, transcript_snapshot: str):
        """Run in a thread. Queries Gemini and primes standby."""
        try:
            system_prompt = f"""You are the brain for a voice AI agent.
Your persona: {self.args.persona}

Analyse the transcript below (what the AI has been saying) and produce a
BOOT_PAYLOAD to load into the next AI node so conversation continues naturally.

Respond ONLY with this exact structure (no markdown, no code blocks):
[SUMMARY] <one sentence: what was being discussed>
[CONTEXT] <key facts/topics the AI should have ready>
[LAST_TOPIC] <the most recent topic or question>
[EMOTION] <tone to match: warm/curious/professional/excited>
[PERSONA] {self.args.persona}"""

            user_prompt = f"""=== TRANSCRIPT (AI speech so far) ===
{transcript_snapshot[-3000:]}
=== END TRANSCRIPT ===

Write the BOOT_PAYLOAD for the next AI node."""

            t0 = time.monotonic()
            response = self.brain.query(system_prompt, user_prompt, max_tokens=250)
            elapsed = time.monotonic() - t0

            print(f"\n[brain] {elapsed:.1f}s → {response[:120]}...")

            # Fire node.prime over the event loop
            if self.loop and response:
                asyncio.run_coroutine_threadsafe(
                    self.send_node_prime(response, self.args.voice or ""),
                    self.loop
                )

        except Exception as e:
            print(f"[brain] ERROR: {e}")
        finally:
            with self._brain_lock:
                self._brain_in_flight = False

    def _maybe_query_brain(self):
        """Called from the receive loop. Fires brain query if conditions met."""
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
        t = threading.Thread(
            target=self._brain_worker,
            args=(self.transcript.get(),),
            daemon=True
        )
        t.start()

    # ── WebSocket message handler ──────────────────────────────────────────

    async def _on_message(self, msg: str):
        try:
            ev = json.loads(msg)
        except Exception:
            return

        t = ev.get("type", "")

        if t == "session.created":
            print(f"  session.created  id={ev.get('session',{}).get('id','?')}")

        elif t == "session.updated":
            print(f"  session.updated")

        elif t == "session.ready":
            print(f"  session.ready — PP nodes loaded, sending audio...")

        elif t == "response.audio.delta":
            b64 = ev.get("delta", "")
            if b64:
                pcm = pcm16_to_float32(b64_to_pcm16(b64))
                try:
                    self._audio_out_q.put_nowait(pcm)
                except queue.Full:
                    pass  # drop if buffer full (prevents stall)

        elif t == "transcript.delta":
            text = ev.get("text", "")
            if text:
                self.transcript.append(text)
                print(f"\r  AI: {self.transcript.get()[-60:]}", end="", flush=True)
                self._maybe_query_brain()

        elif t == "node.status":
            self.node_state = {
                "active":  ev.get("active",  False),
                "standby": ev.get("standby", False),
                "filler":  ev.get("filler",  False),
                "state":   ev.get("state",   ""),
            }
            s = self.node_state
            print(f"\n  [node.status] active={s['active']} standby={s['standby']} "
                  f"filler={s['filler']} state={s['state']}")

        elif t == "node.standby_ready":
            print(f"\n  [node.standby_ready] — switching to standby...")
            self._standby_primed = True
            # Auto-switch to standby at silence
            await self.send_node_switch("standby")

        elif t == "session_error" or t == "error":
            print(f"\n  ERROR: [{ev.get('type','?')}] {ev.get('error',{}).get('message','')}")

        else:
            # Ignore unknown events silently
            pass

    # ── Mic input loop ─────────────────────────────────────────────────────

    async def _mic_sender(self):
        """Read mic frames and send to server every 80ms."""
        if not sd:
            print("  No sounddevice — mic input disabled.")
            await asyncio.sleep(99999)
            return

        mic_q: queue.Queue = queue.Queue()

        def mic_callback(indata, frames, time_info, status):
            mic_q.put_nowait(bytes(indata))

        with sd.RawInputStream(
            samplerate=INPUT_RATE, channels=1, dtype="int16",
            blocksize=FRAME_SAMPLES_IN,
            callback=mic_callback
        ):
            while True:
                try:
                    raw = mic_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                b64 = pcm16_to_b64(raw)
                await self._send({"type": "input_audio_buffer.append", "audio": b64})

    # ── Speaker output loop ────────────────────────────────────────────────

    def _speaker_thread(self):
        """Drain _audio_out_q and play via sounddevice."""
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

    # ── Manual command loop (keyboard) ─────────────────────────────────────

    async def _command_loop(self):
        """Simple keyboard command interface for manual control."""
        cmds = """
  Commands:
    s         — switch to standby (if primed)
    f         — switch to filler (immediate)
    r         — reset filler
    p <text>  — prime standby with custom prompt
    t         — show transcript
    n         — show node status
    q         — quit
"""
        print(cmds)
        loop = asyncio.get_event_loop()
        while True:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
                line = line.strip()
            except Exception:
                break

            if not line:
                continue
            if line == "q":
                print("Quitting...")
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
                prompt = line[2:].strip()
                await self.send_node_prime(prompt)
            elif line == "t":
                print(f"\n--- Transcript ---\n{self.transcript.get()}\n---")
            elif line == "n":
                print(f"\n  Node state: {self.node_state}")
            else:
                print(f"  Unknown command: {line}")

    # ── Session setup ──────────────────────────────────────────────────────

    async def _send_session_update(self):
        """Send session.update with persona and voice."""
        session: dict = {
            "instructions": self.args.persona,
            "filler_prompt": self.args.filler,
        }

        # Encode voice
        if self.args.voice and self.args.voice.upper() in VOICE_NAMES:
            session["text_prompt_tokens"] = encode_voice_tokens(self.args.voice.upper())

        await self._send({"type": "session.update", "session": session})
        print(f"  Sent session.update, waiting for session.ready...")

    # ── Main run ───────────────────────────────────────────────────────────

    async def run(self):
        url = f"ws://{self.args.host}:{self.args.port}/v1/realtime"
        print(f"Connecting to {url} ...")

        # Start speaker thread
        speaker_t = threading.Thread(target=self._speaker_thread, daemon=True)
        speaker_t.start()

        self.loop = asyncio.get_event_loop()

        try:
            async with websockets.connect(url, max_size=16*1024*1024) as ws:
                self.ws = ws
                print("Connected!")

                # Wait for session.created
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                await self._on_message(raw)

                # Send session configuration
                await self._send_session_update()

                # Run mic sender + command loop + receive loop concurrently
                await asyncio.gather(
                    self._receive_loop(),
                    self._mic_sender(),
                    self._command_loop(),
                )

        except websockets.exceptions.ConnectionRefusedError:
            print(f"\nERROR: Connection refused at {url}")
            print("Is the gateway running on the Spark?")
        except asyncio.TimeoutError:
            print("\nERROR: Timeout waiting for server response")
        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            if self.brain:
                self.brain.close()

    async def _receive_loop(self):
        async for msg in self.ws:
            await self._on_message(msg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pick_voice() -> str:
    """Interactive voice selection."""
    groups = [
        ("Natural Female", ["NATF0","NATF1","NATF2","NATF3"]),
        ("Natural Male",   ["NATM0","NATM1","NATM2","NATM3"]),
        ("Varied Female",  ["VARF0","VARF1","VARF2","VARF3","VARF4"]),
        ("Varied Male",    ["VARM0","VARM1","VARM2","VARM3","VARM4"]),
    ]
    print("\n  Available voices:")
    idx = 1
    flat = []
    for group_name, voices in groups:
        print(f"    \u2500\u2500 {group_name} \u2500\u2500")
        for v in voices:
            print(f"    {idx:2d}. {v}")
            flat.append(v)
            idx += 1
    print(f"     0. (no voice prompt \u2014 use default)\n")

    while True:
        try:
            sel = int(input("  Select voice [1-18, 0=default]: ").strip())
            if sel == 0:
                return ""
            if 1 <= sel <= len(flat):
                print(f"  Selected: {flat[sel-1]}")
                return flat[sel-1]
        except (ValueError, EOFError):
            pass


def main():
    p = argparse.ArgumentParser(description="PersonaPlex v3 Brain Client")
    p.add_argument("--host", default="192.168.2.117")
    p.add_argument("--port", type=int, default=8998)
    p.add_argument("--persona", default="You are a helpful, friendly AI assistant.")
    p.add_argument("--filler",  default="Hold on, I'm working on that for you. Just a moment.")
    p.add_argument("--voice",   default="", help="Voice name e.g. NATF0, or blank for interactive")
    p.add_argument("--openrouter-key", default=os.environ.get("OPENROUTER_API_KEY", ""),
                   help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    p.add_argument("--model", default=OPENROUTER_MODEL, help="OpenRouter model to use")
    p.add_argument("--no-brain", action="store_true", help="Disable brain (audio test only)")
    args = p.parse_args()

    print(f"""
PersonaPlex v3 Brain Client
  Server  : {args.host}:{args.port}
  Persona : {args.persona[:60]}
  Filler  : {args.filler[:60]}
  Model   : {args.model if not args.no_brain else 'disabled'}
""")

    # Interactive voice selection if not provided
    if not args.voice:
        args.voice = pick_voice()

    client = PersonaPlexClient(args)
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
