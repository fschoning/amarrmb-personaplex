#!/usr/bin/env python3
"""
PersonaPlex v3 — Workstation Brain Client
==========================================
Connects to the PersonaPlex gateway on the Spark.
  Receives: AI audio (plays to speakers) + transcript text (feeds brain)
  Sends:    Human mic audio + node commands (switch, prime, etc.)

Brain architecture uses Gemini Live API:
  - ONE persistent WebSocket session to Gemini for the whole conversation
  - Transcript deltas streamed in with turn_complete=False (no cost for eval)
  - Periodic lightweight "evaluate" trigger (10 tokens) → streaming response
  - Model holds all context internally — you only pay for NEW tokens

Cost model vs polling:
  Polling:    5000 tokens in + 300 out every 8s  (expensive, grows with transcript)
  Live:       50 tokens in continuously + 10-token eval trigger + 300 out on action
              = ~90% cheaper for long sessions

Usage:
  pip install websockets sounddevice numpy google-genai

  export GOOGLE_API_KEY=AIza...
  python brain_client.py --host 192.168.2.117 --port 8998 \\
      --persona "You are Jane, a curious AI who loves science." \\
      --voice NATF0

  # Without brain (audio-only test):
  python brain_client.py --host 192.168.2.117 --no-brain
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
    from google import genai as google_genai
    from google.genai import types as genai_types
    _GENAI = True
except ImportError:
    _GENAI = False
    print("WARNING: pip install google-genai — brain will be disabled")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_RATE  = 24000
OUTPUT_RATE = 48000
FRAME_MS    = 80
FRAME_SAMPLES_IN  = INPUT_RATE  * FRAME_MS // 1000   # 1920 samples
FRAME_SAMPLES_OUT = OUTPUT_RATE * FRAME_MS // 1000   # 3840 samples

GEMINI_LIVE_MODEL = "gemini-2.0-flash-exp"

# How many NEW chars to accumulate before triggering an evaluation
# (~30 words ≈ 150 chars. Trigger every ~150 new chars.)
EVAL_TRIGGER_CHARS = 150

# Minimum chars in transcript before first eval
BRAIN_MIN_CHARS = 80

# Live session max duration before reconnect (Google limit ~10 min)
LIVE_SESSION_MAX_S = 540   # reconnect at 9 min proactively

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
    """Thread-safe rolling transcript with delta tracking for the Live brain."""

    def __init__(self):
        self._lock = threading.Lock()
        self._full = ""              # full text since session start
        self._sent = 0              # chars already sent to Gemini Live session
        self._session_start = ""    # summary at session reconnect point

    def append(self, delta: str):
        with self._lock:
            self._full += delta
            # Rolling window: keep last 100K chars
            if len(self._full) > 100_000:
                trim = len(self._full) - 80_000
                self._full = self._full[trim:]
                self._sent = max(0, self._sent - trim)

    def unsent(self) -> str:
        """Return text not yet sent to the Live session."""
        with self._lock:
            return self._full[self._sent:]

    def mark_sent(self, n: int):
        """Mark n additional chars as sent."""
        with self._lock:
            self._sent = min(self._sent + n, len(self._full))

    def mark_all_sent(self):
        with self._lock:
            self._sent = len(self._full)

    def get(self) -> str:
        with self._lock:
            return self._full

    def summary_prefix(self) -> str:
        """Prefix to inject on reconnect."""
        with self._lock:
            return self._session_start

    def set_summary_prefix(self, s: str):
        with self._lock:
            self._session_start = s
            self._sent = 0  # reset — will re-send from beginning of new session

    def clear(self):
        with self._lock:
            self._full = ""
            self._sent = 0
            self._session_start = ""

    def __len__(self):
        with self._lock:
            return len(self._full)

# ---------------------------------------------------------------------------
# Gemini Live Brain
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are the brain behind a voice AI agent.
AI Persona: {persona}

You are listening to the transcript of what the AI has been saying in real-time.
The transcript arrives incrementally.

Your ONLY jobs are:
1. Keep listening silently and accumulate context.
2. When you observe that NEW EXTERNAL INFORMATION is needed to answer
   a question or continue the conversation well, respond with:

   ACTION_NEEDED
   [REASON] <why new info is needed — 1 sentence>
   [TOPIC] <the specific question or topic requiring new info>
   [BOOT_PAYLOAD]
   [SUMMARY] <what the conversation has been about>
   [CONTEXT] <key facts the AI should know>
   [LAST_TOPIC] <most recent topic being addressed>
   [EMOTION] <tone: warm/curious/professional/excited>
   [PERSONA] {persona}

3. If nothing new is needed, respond with exactly: CONTINUE

Rules:
- Be decisive. If the AI is handling the conversation fine, say CONTINUE.
- Only say ACTION_NEEDED when the user has asked something the AI cannot
  answer without new information or a tool call.
- Do not add commentary. Just CONTINUE or ACTION_NEEDED + structured block.
"""

EVAL_PROMPT = "Evaluate the transcript so far. Action needed? Respond with CONTINUE or ACTION_NEEDED."


class GeminiLiveBrain:
    """
    Maintains a persistent Gemini Live WebSocket session for the entire
    conversation. Transcript deltas are fed continuously. Periodic evaluation
    triggers ask the model to assess whether new information is needed.

    Architecture:
      - One background asyncio task manages the Live session
      - Transcript deltas arrive via queue and are sent with turn_complete=False
      -_evaluate() sends the eval prompt with turn_complete=True
      - Streaming response is parsed for CONTINUE / ACTION_NEEDED
    """

    def __init__(self, api_key: str, persona: str,
                 model: str = GEMINI_LIVE_MODEL,
                 on_action=None):
        self._api_key   = api_key
        self._persona   = persona
        self._model     = model
        self._on_action = on_action   # async callback(boot_payload: str)

        self._system_prompt = SYSTEM_PROMPT_TEMPLATE.format(persona=persona)
        self._client = google_genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1alpha"},
        )
        self._config = genai_types.LiveConnectConfig(
            response_modalities=["TEXT"],
            system_instruction=genai_types.Content(
                parts=[genai_types.Part(text=self._system_prompt)]
            ),
        )

        # Communication with the live loop
        self._delta_q: asyncio.Queue = None   # created in async context
        self._eval_event: asyncio.Event = None
        self._stop_event: asyncio.Event = None
        self._session_task: asyncio.Task = None

        self._session_start_t = 0.0
        self._chars_since_eval = 0
        self._evaluating = False

        print(f"  [brain] Gemini Live session (model={model})")
        print(f"  [brain] Eval trigger every ~{EVAL_TRIGGER_CHARS} new chars")

    async def start(self):
        """Start the Live session manager. Call once from the async context."""
        self._delta_q   = asyncio.Queue()
        self._eval_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._session_task = asyncio.create_task(self._session_loop())

    async def stop(self):
        if self._stop_event:
            self._stop_event.set()
        if self._session_task:
            self._session_task.cancel()
            try:
                await self._session_task
            except asyncio.CancelledError:
                pass

    def feed_transcript(self, delta: str):
        """
        Called with each transcript.delta from the server.
        Non-blocking — just enqueues for the live session.
        """
        if self._delta_q:
            try:
                self._delta_q.put_nowait(("transcript", delta))
            except asyncio.QueueFull:
                pass
        self._chars_since_eval += len(delta)
        # Auto-trigger eval when enough new text has arrived
        if (self._chars_since_eval >= EVAL_TRIGGER_CHARS
                and not self._evaluating
                and self._eval_event):
            self._eval_event.set()

    def force_evaluate(self):
        """Manually trigger an evaluation (keyboard command)."""
        if self._eval_event:
            self._eval_event.set()

    async def _session_loop(self):
        """
        Main loop: manages the Live WebSocket connection, reconnects on
        timeout/error, feeds a summary on reconnect.
        """
        summary = ""   # grows as sessions complete

        while not self._stop_event.is_set():
            try:
                summary = await self._run_session(summary)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"\n  [brain] Live session error: {e} — reconnecting in 3s...")
                await asyncio.sleep(3)

    async def _run_session(self, prior_summary: str) -> str:
        """
        Open one Live session. Returns a summary string when the session
        ends (for the next session to use as context).
        """
        self._session_start_t = time.monotonic()
        self._chars_since_eval = 0
        accumulated_response = ""

        async with self._client.aio.live.connect(
                model=self._model, config=self._config) as session:

            print(f"\n  [brain] Live session connected.")

            # If reconnecting, inject the prior summary first
            if prior_summary:
                await session.send(
                    input=f"[CONTEXT FROM PREVIOUS SESSION]\n{prior_summary}\n"
                           "[END CONTEXT]\n",
                    end_of_turn=False
                )
                print(f"  [brain] Injected prior session summary.")

            # Run sender and receiver concurrently
            sender_task   = asyncio.create_task(
                self._sender(session))
            receiver_task = asyncio.create_task(
                self._receiver(session))

            try:
                done, pending = await asyncio.wait(
                    [sender_task, receiver_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
            except Exception:
                pass

        # Generate a compact summary of this session for continuity
        return await self._compact_summary(session)

    async def _sender(self, session):
        """
        Pull deltas from _delta_q and send with turn_complete=False.
        When _eval_event fires, send the evaluation prompt with turn_complete=True.
        Reconnect if session runs too long.
        """
        while True:
            # Check if session should be renewed
            if time.monotonic() - self._session_start_t > LIVE_SESSION_MAX_S:
                print("\n  [brain] Session age limit — reconnecting for fresh context.")
                return  # causes _run_session to end and restart

            # Drain any pending transcript deltas
            flushed = False
            while not self._delta_q.empty():
                kind, text = await self._delta_q.get()
                if kind == "transcript":
                    await session.send(input=text, end_of_turn=False)
                    flushed = True

            # Check if eval was triggered
            if self._eval_event.is_set():
                self._eval_event.clear()
                self._chars_since_eval = 0
                self._evaluating = True
                print("\n  [brain] Evaluating...", end="", flush=True)
                await session.send(input=EVAL_PROMPT, end_of_turn=True)

            await asyncio.sleep(0.05)

    async def _receiver(self, session):
        """
        Receive streaming tokens from Gemini. Parse for CONTINUE / ACTION_NEEDED.
        """
        response_buf = ""
        async for chunk in session.receive():
            if self._stop_event.is_set():
                break

            text = getattr(chunk, "text", None)
            if text:
                response_buf += text
                print(text, end="", flush=True)

            # Check if the turn is complete
            sc = getattr(chunk, "server_content", None)
            turn_done = sc and getattr(sc, "turn_complete", False)

            if turn_done and response_buf.strip():
                await self._handle_response(response_buf.strip())
                response_buf = ""
                self._evaluating = False
                print()  # newline after response

    async def _handle_response(self, response: str):
        """Parse Gemini's evaluation response and take action."""
        if response.startswith("CONTINUE"):
            print("\n  [brain] ✓ CONTINUE — no action needed")
            return

        if "ACTION_NEEDED" in response:
            print("\n  [brain] ⚡ ACTION NEEDED")
            # Extract the BOOT_PAYLOAD section
            boot_start = response.find("[BOOT_PAYLOAD]")
            if boot_start >= 0:
                boot_payload = response[boot_start + len("[BOOT_PAYLOAD]"):].strip()
            else:
                boot_payload = response

            if self._on_action:
                await self._on_action(boot_payload)
        else:
            # Unexpected response — log it
            print(f"\n  [brain] Unexpected response: {response[:80]}")

    async def _compact_summary(self, session) -> str:
        """
        At session end, ask Gemini for a compact summary to carry forward.
        Uses a new one-shot call (not the Live session) to avoid state issues.
        """
        try:
            model = google_genai.GenerativeModel(
                model_name=self._model,
            )
            resp = model.generate_content(
                (f"Summarize this conversation context in 200 words for handoff "
                 f"to a new session. Focus on key topics, facts, and the last question."),
                generation_config={"max_output_tokens": 250, "temperature": 0.1},
            )
            summary = resp.text.strip()
            print(f"\n  [brain] Session summary: {summary[:80]}...")
            return summary
        except Exception as e:
            print(f"\n  [brain] Compact summary failed: {e}")
            return ""

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

        # Brain (Gemini Live)
        self.brain: GeminiLiveBrain = None
        if not args.no_brain and _GENAI and args.google_api_key:
            self.brain = GeminiLiveBrain(
                api_key=args.google_api_key,
                persona=args.persona,
                model=args.model,
                on_action=self._on_brain_action,
            )
        elif not args.no_brain:
            if not _GENAI:
                print("  Brain: disabled — pip install google-genai")
            else:
                print("  Brain: disabled — set GOOGLE_API_KEY or --google-api-key")
        else:
            print("  Brain: disabled (--no-brain)")

    # ── Brain action callback ─────────────────────────────────────────────

    async def _on_brain_action(self, boot_payload: str):
        """
        Called when Gemini decides ACTION_NEEDED.
        1. Immediately switch to Filler (human hears "hold on...")
        2. Prime Standby with the new boot payload
        3. When standby is ready, switch from Filler → Standby
        """
        print("\n  [brain] ACTION: switching to filler + priming standby...")

        # Step 1: Filler takes over immediately
        await self.send_node_switch("filler")

        # Step 2: Prime standby with the boot payload from brain
        await self.send_node_prime(boot_payload)

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
            print("  session.ready ✓ — PP active, audio flowing...")
            # Start the Gemini Live brain session now that PP is ready
            if self.brain:
                await self.brain.start()
                print("  [brain] Live session starting...")

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
                # Feed to Gemini Live brain (non-blocking)
                if self.brain and len(self.transcript) >= BRAIN_MIN_CHARS:
                    self.brain.feed_transcript(text)

        elif t == "node.status":
            self.node_state = {k: ev.get(k) for k in
                               ("active","standby","filler","state")}
            s = self.node_state
            print(f"\n  [node.status] active={s['active']} standby={s['standby']} "
                  f"filler={s['filler']} state={s['state']}")

        elif t == "node.standby_ready":
            print("\n  [node.standby_ready] — switching to standby...")
            # If filler is active, this is the ACTION flow completing:
            # filler was inserted, standby now primed → switch standby in
            await self.send_node_switch("standby")
            # Tell filler to reset (so it's fresh for next trigger)
            await self.send_node_stop("filler")

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
                    "type": "input_audio_buffer.append",
                    "audio": pcm16_to_b64(raw),
                })

    # ── Speaker output ────────────────────────────────────────────────────

    def _speaker_thread(self):
        if not sd:
            return
        stream = sd.OutputStream(
            samplerate=OUTPUT_RATE, channels=1, dtype="float32")
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
    e        — force brain evaluation now
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
            elif line == "e":
                if self.brain:
                    self.brain.force_evaluate()
                    print("  [brain] Evaluation triggered.")
                else:
                    print("  Brain not running.")
            elif line == "t":
                print(f"\n--- Transcript ---\n{self.transcript.get()}\n---")
            elif line == "n":
                print(f"\n  Node state: {self.node_state}")
            else:
                print(f"  Unknown: {line!r}")

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
                await self.brain.stop()

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
    p.add_argument("--persona",
                   default="You are a helpful, friendly AI assistant.")
    p.add_argument("--filler",
                   default="Hold on, I'm working on that for you. Just a moment.")
    p.add_argument("--voice", default="",
                   help="Voice name e.g. NATF0, or blank for interactive")
    p.add_argument("--google-api-key",
                   default=os.environ.get("GOOGLE_API_KEY", ""),
                   dest="google_api_key",
                   help="Google AI API key (or set GOOGLE_API_KEY env var)")
    p.add_argument("--model", default=GEMINI_LIVE_MODEL)
    p.add_argument("--no-brain", action="store_true")
    args = p.parse_args()

    print(f"""
PersonaPlex v3 Brain Client
  Server  : {args.host}:{args.port}
  Persona : {args.persona[:60]}
  Filler  : {args.filler[:60]}
  Brain   : {'disabled' if args.no_brain else
             f'Gemini Live ({args.model}) — streaming, event-driven'}
""")

    if not args.voice:
        args.voice = pick_voice()

    try:
        asyncio.run(PersonaPlexClient(args).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
