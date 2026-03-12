#!/usr/bin/env python3
"""
PersonaPlex v2 — Test Client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Connects to the PersonaPlex gateway via WebSocket, sends audio, receives audio.
Runs on Windows/Linux/Mac — requires only: pip install websockets sounddevice numpy

Usage:
  # Interactive mic mode (default):
  python test_client.py --host gx10-8c8b --port 8998

  # Send a .wav file:
  python test_client.py --host gx10-8c8b --port 8998 --wav test.wav

  # Loopback test (send silence, verify protocol flow):
  python test_client.py --host gx10-8c8b --port 8998 --loopback

  # With system prompt:
  python test_client.py --host gx10-8c8b --prompt "You are a helpful assistant."
"""

import argparse
import asyncio
import base64
import json
import queue
import struct
import sys
import threading
import time
import wave
from pathlib import Path

try:
    import websockets
except ImportError:
    print("ERROR: pip install websockets")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    np = None
    print("WARNING: numpy not installed — WAV file mode unavailable")

try:
    import sounddevice as sd
except ImportError:
    sd = None
    print("WARNING: sounddevice not installed — mic/speaker mode unavailable")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
INPUT_RATE = 24000   # What the server expects
OUTPUT_RATE = 24000  # What the server sends back (direct from Mimi decoder)
FRAME_MS = 80        # Send audio every 80ms
FRAME_SAMPLES = INPUT_RATE * FRAME_MS // 1000  # 1920

def pcm16_to_b64(samples_int16: bytes) -> str:
    """Encode raw PCM16 LE bytes to base64 string."""
    return base64.b64encode(samples_int16).decode("ascii")

def b64_to_pcm16(b64: str) -> bytes:
    """Decode base64 string to raw PCM16 LE bytes."""
    return base64.b64decode(b64)

def float32_to_pcm16_bytes(arr) -> bytes:
    """Convert numpy float32 array to PCM16 LE bytes."""
    clamped = np.clip(arr, -1.0, 1.0)
    int16 = (clamped * 32767).astype(np.int16)
    return int16.tobytes()

def pcm16_bytes_to_float32(data: bytes):
    """Convert PCM16 LE bytes to numpy float32 array."""
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class PersonaPlexClient:
    def __init__(self, url: str, prompt: str = "", voice_prompt_path: str = None):
        self.url = url
        self.prompt = prompt
        self.voice_prompt_path = voice_prompt_path
        self.ws = None
        self.session_id = None
        self.session_ready = asyncio.Event()
        self.connected = asyncio.Event()
        self.running = True
        self.audio_received = 0
        self.frames_sent = 0
        self.output_buffer = bytearray()

        # Streaming speaker output with pre-buffering to avoid clicks
        self._speaker_queue = queue.Queue()
        self._speaker_stream = None
        self._speaker_started = False
        self._prebuffer_count = 3  # wait for N chunks before starting playback
        self._prebuffer = []
        if sd and np:
            try:
                self._speaker_stream = sd.OutputStream(
                    samplerate=OUTPUT_RATE,
                    channels=1,
                    dtype='float32',
                    blocksize=2048,   # larger blocks = fewer callbacks = fewer underruns
                    callback=self._speaker_callback,
                )
                self._speaker_stream.start()
            except Exception as e:
                print(f"  [speaker] Could not open output stream: {e}")
                self._speaker_stream = None

    def _speaker_callback(self, outdata, frames, time_info, status):
        """Sounddevice output callback — pulls audio from the queue."""
        filled = 0
        while filled < frames:
            try:
                chunk = self._speaker_queue.get_nowait()
                n = min(len(chunk), frames - filled)
                outdata[filled:filled + n, 0] = chunk[:n]
                filled += n
                if n < len(chunk):
                    # Put remainder back (front of queue)
                    self._speaker_queue.put(chunk[n:])
            except queue.Empty:
                # No more audio — fill rest with silence
                outdata[filled:, 0] = 0
                break

    async def connect(self):
        print(f"Connecting to {self.url} ...")
        self.ws = await websockets.connect(
            self.url,
            max_size=16 * 1024 * 1024,
            ping_interval=30,
            ping_timeout=60,
        )
        self.connected.set()
        print("Connected!")

    async def send_session_update(self):
        """Send session.update to configure the session."""
        session_cfg = {
            "instructions": self.prompt or "You are a helpful voice assistant.",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "temperature": 0.8,
            "top_k": 250,
        }

        # Attach voice prompt if provided
        if self.voice_prompt_path and Path(self.voice_prompt_path).exists():
            with open(self.voice_prompt_path, "rb") as f:
                vp_bytes = f.read()
            session_cfg["voice_prompt_embedding"] = base64.b64encode(vp_bytes).decode()
            print(f"  Voice prompt: {self.voice_prompt_path} ({len(vp_bytes)} bytes)")

        msg = json.dumps({"type": "session.update", "session": session_cfg})
        await self.ws.send(msg)
        print("Sent session.update, waiting for session.ready...")

    async def receive_loop(self):
        """Receive and process server events."""
        try:
            async for raw in self.ws:
                if not self.running:
                    break
                event = json.loads(raw)
                etype = event.get("type", "")

                if etype == "session.created":
                    self.session_id = event.get("session", {}).get("id", "?")
                    print(f"  session.created  id={self.session_id}")

                elif etype == "session.updated":
                    print(f"  session.updated")

                elif etype == "session.ready":
                    print(f"  session.ready — you can now speak!")
                    self.session_ready.set()

                elif etype == "response.audio.delta":
                    b64 = event.get("delta", "")
                    pcm_bytes = b64_to_pcm16(b64)
                    self.audio_received += len(pcm_bytes)
                    self.output_buffer.extend(pcm_bytes)

                    # Stream to speaker via queue (continuous, no restarts)
                    if self._speaker_stream and np:
                        samples = pcm16_bytes_to_float32(pcm_bytes)
                        if not self._speaker_started:
                            # Pre-buffer to avoid initial underrun clicks
                            self._prebuffer.append(samples)
                            if len(self._prebuffer) >= self._prebuffer_count:
                                for buf in self._prebuffer:
                                    self._speaker_queue.put(buf)
                                self._prebuffer.clear()
                                self._speaker_started = True
                        else:
                            self._speaker_queue.put(samples)

                    # Progress indicator
                    ms = (self.audio_received // 2) * 1000 // OUTPUT_RATE
                    print(f"\r  Received {ms}ms of audio ({self.audio_received} bytes)", end="", flush=True)

                elif etype == "response.audio.done":
                    print(f"\n  response.audio.done")

                elif etype == "response.text.delta":
                    delta = event.get("delta", "")
                    print(f"  text: {delta}", end="", flush=True)

                elif etype == "error":
                    err = event.get("error", {})
                    print(f"\n  ERROR: [{err.get('type')}] {err.get('message')}")

                else:
                    print(f"  {etype}: {json.dumps(event)[:120]}")

        except websockets.exceptions.ConnectionClosedOK:
            print("\nConnection closed normally.")
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"\nConnection closed with error: {e}")
        except Exception as e:
            print(f"\nReceive error: {e}")
        finally:
            self.running = False

    async def send_silence(self, duration_s: float = 5.0):
        """Send silence frames (loopback test)."""
        await self.session_ready.wait()
        total_frames = int(duration_s * 1000 / FRAME_MS)
        silence = b'\x00' * (FRAME_SAMPLES * 2)  # 1920 samples * 2 bytes

        print(f"Sending {duration_s}s of silence ({total_frames} frames)...")
        for i in range(total_frames):
            if not self.running:
                break
            msg = json.dumps({
                "type": "input_audio_buffer.append",
                "audio": pcm16_to_b64(silence),
            })
            await self.ws.send(msg)
            self.frames_sent += 1
            await asyncio.sleep(FRAME_MS / 1000.0)

        print(f"Sent {self.frames_sent} frames.")

    async def send_wav_file(self, wav_path: str):
        """Read a WAV file and stream it frame by frame."""
        if not np:
            print("ERROR: numpy required for WAV mode. pip install numpy")
            return

        await self.session_ready.wait()

        print(f"Reading {wav_path}...")
        with wave.open(wav_path, 'rb') as wf:
            assert wf.getnchannels() == 1, f"Expected mono, got {wf.getnchannels()} channels"
            raw = wf.readframes(wf.getnframes())
            rate = wf.getframerate()
            width = wf.getsampwidth()

        # Convert to float32
        if width == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            print(f"ERROR: Unsupported sample width {width}")
            return

        # Resample to 24kHz if needed
        if rate != INPUT_RATE:
            print(f"  Resampling {rate} Hz → {INPUT_RATE} Hz...")
            duration = len(samples) / rate
            n_out = int(duration * INPUT_RATE)
            indices = np.linspace(0, len(samples) - 1, n_out)
            samples = np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)

        total_samples = len(samples)
        total_frames = total_samples // FRAME_SAMPLES
        print(f"  {total_samples} samples, {total_frames} frames, "
              f"{total_samples / INPUT_RATE:.1f}s at {INPUT_RATE} Hz")

        for i in range(total_frames):
            if not self.running:
                break
            chunk = samples[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            pcm_bytes = float32_to_pcm16_bytes(chunk)
            msg = json.dumps({
                "type": "input_audio_buffer.append",
                "audio": pcm16_to_b64(pcm_bytes),
            })
            await self.ws.send(msg)
            self.frames_sent += 1
            print(f"\r  Sent frame {self.frames_sent}/{total_frames}", end="", flush=True)
            await asyncio.sleep(FRAME_MS / 1000.0)  # Real-time pacing

        print(f"\nDone sending {self.frames_sent} frames.")

    async def send_mic(self):
        """Capture from microphone and stream in real-time."""
        if not sd or not np:
            print("ERROR: sounddevice and numpy required for mic mode.")
            print("  pip install sounddevice numpy")
            return

        await self.session_ready.wait()

        print(f"Streaming from microphone at {INPUT_RATE} Hz...")
        print("  Press Ctrl+C to stop.\n")

        loop = asyncio.get_event_loop()
        audio_queue = asyncio.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"  [mic status: {status}]")
            # indata is float32 [frames, channels]
            pcm_bytes = float32_to_pcm16_bytes(indata[:, 0])
            loop.call_soon_threadsafe(audio_queue.put_nowait, pcm_bytes)

        stream = sd.InputStream(
            samplerate=INPUT_RATE,
            blocksize=FRAME_SAMPLES,
            channels=1,
            dtype='float32',
            callback=audio_callback,
        )

        with stream:
            while self.running:
                try:
                    pcm_bytes = await asyncio.wait_for(audio_queue.get(), timeout=1.0)
                    msg = json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": pcm16_to_b64(pcm_bytes),
                    })
                    await self.ws.send(msg)
                    self.frames_sent += 1
                except asyncio.TimeoutError:
                    continue
                except KeyboardInterrupt:
                    break

        print(f"\nMic stopped. Sent {self.frames_sent} frames.")

    async def save_output(self, path: str):
        """Save received audio to a WAV file."""
        if not self.output_buffer:
            print("No audio received to save.")
            return
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(OUTPUT_RATE)
            wf.writeframes(bytes(self.output_buffer))
        ms = len(self.output_buffer) // 2 * 1000 // OUTPUT_RATE
        print(f"Saved {ms}ms of output audio to {path}")

    async def close(self):
        if self._speaker_stream:
            # Drain remaining audio before closing
            await asyncio.sleep(0.5)
            self._speaker_stream.stop()
            self._speaker_stream.close()
            self._speaker_stream = None
        if self.ws:
            await self.ws.close()


# ---------------------------------------------------------------------------
# Voice selection helpers
# ---------------------------------------------------------------------------
def find_voice_prompts(voices_dir: str) -> list[Path]:
    """Find all .pt voice prompt files in a directory."""
    vdir = Path(voices_dir)
    if not vdir.exists():
        return []
    return sorted(vdir.glob("*.pt"))


def interactive_voice_select(voices_dir: str) -> str | None:
    """Show a numbered list of voices and let the user pick one."""
    voices = find_voice_prompts(voices_dir)
    if not voices:
        print(f"  No .pt voice files found in {voices_dir}")
        return None

    print(f"\n  Available voices ({len(voices)}):")
    for i, v in enumerate(voices, 1):
        print(f"    {i:2d}. {v.stem}")
    print(f"    {0:2d}. (no voice prompt — use default)")

    while True:
        try:
            choice = input(f"\n  Select voice [1-{len(voices)}, 0=default]: ").strip()
            if not choice:
                idx = 1  # default to first voice
            else:
                idx = int(choice)
            if idx == 0:
                return None
            if 1 <= idx <= len(voices):
                selected = voices[idx - 1]
                print(f"  Selected: {selected.stem}")
                return str(selected)
            print(f"  Please enter 0-{len(voices)}")
        except (ValueError, EOFError):
            return str(voices[0]) if voices else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description="PersonaPlex v2 Test Client")
    parser.add_argument("--host", default="localhost", help="Gateway hostname/IP")
    parser.add_argument("--port", type=int, default=8998, help="Gateway port")
    parser.add_argument("--prompt", default="", help="System prompt / instructions")
    parser.add_argument("--voice-prompt", default=None, help="Path to .pt voice prompt file")
    parser.add_argument("--voices-dir", default=None,
                        help="Directory containing .pt voice files (enables interactive picker)")
    parser.add_argument("--wav", default=None, help="Path to .wav file to send (mono)")
    parser.add_argument("--loopback", action="store_true", help="Send silence (protocol test)")
    parser.add_argument("--duration", type=float, default=10.0, help="Loopback/mic duration (s)")
    parser.add_argument("--save", default=None, help="Save output audio to .wav file")
    parser.add_argument("--tls", action="store_true", help="Use wss:// instead of ws://")
    args = parser.parse_args()

    # Interactive voice selection
    if args.voices_dir and not args.voice_prompt:
        args.voice_prompt = interactive_voice_select(args.voices_dir)

    scheme = "wss" if args.tls else "ws"
    url = f"{scheme}://{args.host}:{args.port}/v1/realtime"

    client = PersonaPlexClient(url, args.prompt, args.voice_prompt)

    try:
        await client.connect()

        # Start receiver in background
        recv_task = asyncio.create_task(client.receive_loop())

        # Wait for session.created, then send config
        await asyncio.sleep(0.5)
        await client.send_session_update()

        # Choose input mode
        if args.wav:
            await client.send_wav_file(args.wav)
            # Wait a bit for remaining responses
            print("Waiting for remaining audio responses...")
            await asyncio.sleep(5)
        elif args.loopback:
            await client.send_silence(args.duration)
            await asyncio.sleep(3)
        else:
            # Mic mode
            if sd and np:
                await client.send_mic()
            else:
                print("\nNo input mode selected and sounddevice not available.")
                print("Use --wav <file> or --loopback, or install sounddevice:")
                print("  pip install sounddevice numpy")
                await asyncio.sleep(2)

        # Save output if requested
        if args.save:
            await client.save_output(args.save)

        # Print summary
        print(f"\n--- Summary ---")
        print(f"  Frames sent:     {client.frames_sent}")
        print(f"  Audio received:  {client.audio_received} bytes "
              f"({client.audio_received // 2 * 1000 // OUTPUT_RATE if client.audio_received else 0} ms)")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except ConnectionRefusedError:
        print(f"\nERROR: Connection refused at {url}")
        print(f"Is the gateway running on {args.host}:{args.port}?")
    except Exception as e:
        print(f"\nERROR: {e}")
    finally:
        client.running = False
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
