# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT
#
# Triton Python Backend — PersonaPlex Monolithic Pipeline
#
# Mirrors the original server.py architecture: ONE process, ONE mimi,
# ONE LM — all direct function calls, zero IPC overhead.
#
# Frame flow:
#   START=1  → reset streaming, load prompts, run system conditioning
#   regular  → encode(pcm) → lm.step(codes) → decode(tokens) → PCM out
#   END=1    → cleanup
#
# One Triton instance handles one session (sequence batcher guarantees this).

import os
import tempfile
import time
import threading
from typing import Optional

import numpy as np
import sentencepiece
import torch

# CPU DSP resampling — polyphase filtering to smooth Mimi frame transitions
# soxr is preferred (libsoxr, ARM NEON-optimized), scipy as fallback
try:
    import soxr
    _RESAMPLE = "soxr"
except ImportError:
    soxr = None
    try:
        from scipy.signal import resample_poly
        _RESAMPLE = "scipy"
    except ImportError:
        resample_poly = None
        _RESAMPLE = "linear"

import triton_python_backend_utils as pb_utils

from huggingface_hub import hf_hub_download
from moshi.models import loaders
from moshi.models.lm import LMGen
from moshi.fp8_quantize import quantize_model

_HF_REPO      = os.environ.get("HF_REPO", "") or loaders.DEFAULT_REPO
_DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_USE_FP8      = os.environ.get("PERSONAPLEX_FP8", "1") == "1"
_TEMPERATURE  = float(os.environ.get("LM_TEMPERATURE", "0.8"))
_TOP_K        = int(os.environ.get("LM_TOP_K", "250"))


class TritonPythonModel:

    def initialize(self, args: dict):
        self.logger = pb_utils.Logger
        self.logger.log_info("personaplex_pipeline: loading models...")

        # --- Load Mimi (single instance for encode + decode) ---
        mimi_weight = hf_hub_download(_HF_REPO, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, _DEVICE)
        if _USE_FP8:
            self.mimi = self.mimi.half()
            self.mimi.torch_compile_encoder_decoder = True
            self.mimi = torch.compile(self.mimi)
            self.logger.log_info("personaplex_pipeline: Mimi loaded (FP16 + compiled)")
        else:
            self.logger.log_info("personaplex_pipeline: Mimi loaded")
        self.mimi.streaming_forever(1)
        self._input_dtype = torch.float16 if _USE_FP8 else torch.float32

        # --- Load LM ---
        self.logger.log_info("personaplex_pipeline: loading LM weights...")
        lm_weight = hf_hub_download(_HF_REPO, loaders.MOSHI_NAME)
        lm = loaders.get_moshi_lm(lm_weight, device=_DEVICE)
        lm.eval()

        if _USE_FP8:
            self.logger.log_info("personaplex_pipeline: applying FP8 quantisation...")
            quantize_model(lm)

        self.lm_gen = LMGen(
            lm,
            device=_DEVICE,
            temp=_TEMPERATURE,
            top_k=_TOP_K,
        )
        self.lm_gen.streaming_forever(1)

        # --- Load text tokenizer ---
        tokenizer_path = hf_hub_download(_HF_REPO, loaders.TEXT_TOKENIZER_NAME)
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

        # --- Warmup: 4 rounds of encode → step → decode (mirrors server.py) ---
        self.logger.log_info("personaplex_pipeline: warming up (4 rounds)...")
        frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        for _ in range(4):
            chunk = torch.zeros(1, 1, frame_size, dtype=self._input_dtype, device=_DEVICE)
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c:c + 1])
                if tokens is not None:
                    _ = self.mimi.decode(tokens[:, 1:9])
        if _DEVICE.type == "cuda":
            torch.cuda.synchronize()
        self.logger.log_info("personaplex_pipeline: warmup done.")

        # Free BF16 inproj weights after warmup
        if _USE_FP8:
            try:
                from moshi.fp8_quantize import free_bf16_inproj
                free_bf16_inproj(lm)
                self.logger.log_info("personaplex_pipeline: freed BF16 inproj weights")
            except ImportError:
                pass

        # Session state
        self._voice_prompt_bytes: Optional[bytes] = None
        self._text_prompt_tokens: Optional[list]  = None
        self._active = False
        self._frame_count = 0
        self._text_tokens: list = []          # accumulated text token IDs
        self._last_decoded_len: int = 0       # chars already output

        self.logger.log_info("personaplex_pipeline: ready.")

    # ------------------------------------------------------------------
    # Voice name resolution
    # ------------------------------------------------------------------

    def _resolve_voice_name(self, name: str) -> str | None:
        """Resolve a voice name (e.g. 'NATF0') to the .pt file path in HF cache."""
        # Look in the HF cache for the PersonaPlex voices directory
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        voices_base = os.path.join(hf_home, "hub")

        # Search for the voices directory in any PersonaPlex snapshot
        for root, dirs, files in os.walk(voices_base):
            if os.path.basename(root) == "voices":
                pt_path = os.path.join(root, f"{name}.pt")
                if os.path.exists(pt_path):
                    return pt_path

        # Also check a direct path
        direct = os.path.join(voices_base, "voices", f"{name}.pt")
        if os.path.exists(direct):
            return direct

        return None

    # ------------------------------------------------------------------
    # System prompt conditioning
    # ------------------------------------------------------------------

    def _run_system_prompts(self):
        self.mimi.reset_streaming()

        # Voice loading: name-based (from tokens) or binary (from voice_prompt_bytes)
        if hasattr(self, '_voice_name') and self._voice_name:
            voice_path = self._resolve_voice_name(self._voice_name)
            if voice_path:
                self.logger.log_info(f"personaplex_pipeline: loading voice '{self._voice_name}'")
                self.lm_gen.load_voice_prompt_embeddings(voice_path)
            else:
                self.logger.log_info(f"personaplex_pipeline: voice '{self._voice_name}' not found")
        elif self._voice_prompt_bytes:
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                f.write(self._voice_prompt_bytes)
                tmp_path = f.name
            self.lm_gen.load_voice_prompt_embeddings(tmp_path)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if self._text_prompt_tokens:
            self.lm_gen.text_prompt_tokens = self._text_prompt_tokens
        else:
            # Must be an iterable (not None) — _step_text_prompt_core iterates it
            self.lm_gen.text_prompt_tokens = []

        self.lm_gen.step_system_prompts(self.mimi)

        # CRITICAL: reset mimi streaming AFTER system prompts (matches server.py line 346)
        # System prompts leave residual state in mimi's streaming buffers
        self.mimi.reset_streaming()

        # Post-system-prompt warmup: eat torch.compile re-trace cost HERE
        # instead of during real-time frames. reset_streaming() invalidates
        # torch.compile guards, so the first encode/decode after reset triggers
        # a re-trace (~2-4s). Run it now during session setup.
        frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        for _ in range(4):
            chunk = torch.zeros(1, 1, frame_size, dtype=self._input_dtype, device=_DEVICE)
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c:c + 1])
                if tokens is not None:
                    _ = self.mimi.decode(tokens[:, 1:9])
        if _DEVICE.type == "cuda":
            torch.cuda.synchronize()

        # Reset again to clear warmup residue before real conversation
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()

    # ------------------------------------------------------------------
    # Main execute — ALL processing in one function call, zero IPC
    # ------------------------------------------------------------------

    @torch.no_grad()
    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                resp = self._process_one(request)
                responses.append(resp)
            except Exception as e:
                self.logger.log_error(f"personaplex_pipeline error: {e}")
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                ))

        return responses

    def _process_one(self, request):
        start = bool(
            pb_utils.get_input_tensor_by_name(request, "START")
            .as_numpy().flat[0] > 0.5
        )
        end = bool(
            pb_utils.get_input_tensor_by_name(request, "END")
            .as_numpy().flat[0] > 0.5
        )

        # --- END ---
        if end:
            self.lm_gen.reset_streaming()
            self.mimi.reset_streaming()
            self._active = False
            return self._make_response(
                pcm=np.zeros((1, 1, 1920), dtype=np.float32),
                text_token=np.array([0], dtype=np.int32),
                ready=np.array([False]),
            )

        # --- START ---
        if start:
            self.lm_gen.reset_streaming()
            self._active = True
            self._frame_count = 0

            # Extract voice prompt (binary .pt via voice_prompt_embedding)
            vp_tensor = pb_utils.get_input_tensor_by_name(request, "VOICE_PROMPT_BYTES")
            if vp_tensor is not None:
                raw = vp_tensor.as_numpy()
                if raw.size > 0:
                    self._voice_prompt_bytes = raw.flat[0]

            # Extract text prompt tokens — supports dual-sentinel encoding:
            #   [-999, a,s,c,i,i...]          → voice name only
            #   [tok1, tok2, ...]              → text tokens only
            #   [-999, a,s,c,i,i..., -998, tok1, tok2, ...] → voice name + text tokens
            # Sentinel -999: voice name (ASCII int32 chars follow)
            # Sentinel -998: separator between voice name and text tokens
            self._voice_name = None
            tp_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT_PROMPT_TOKENS")
            if tp_tensor is not None:
                tok = tp_tensor.as_numpy().flatten()
                if tok.size > 1 and tok[0] == -999:
                    # Find -998 separator if present
                    sep_idx = None
                    for i, v in enumerate(tok):
                        if v == -998:
                            sep_idx = i
                            break
                    if sep_idx is not None:
                        # Dual: voice name up to separator, text tokens after
                        self._voice_name = "".join(
                            chr(c) for c in tok[1:sep_idx] if 32 <= c < 128)
                        rest = tok[sep_idx + 1:]
                        self._text_prompt_tokens = rest.tolist() if rest.size > 0 else None
                        self.logger.log_info(
                            f"personaplex_pipeline: voice='{self._voice_name}' "
                            f"+ {len(self._text_prompt_tokens or [])} text tokens")
                    else:
                        # Voice name only (legacy / no text instruction)
                        self._voice_name = "".join(
                            chr(c) for c in tok[1:] if 32 <= c < 128)
                        self._text_prompt_tokens = None
                        self.logger.log_info(
                            f"personaplex_pipeline: voice name from tokens: '{self._voice_name}'")
                elif tok.size > 0:
                    self._text_prompt_tokens = tok.tolist()
                else:
                    self._text_prompt_tokens = None


            self.logger.log_info("personaplex_pipeline: running system prompts...")
            self._run_system_prompts()
            self.logger.log_info("personaplex_pipeline: system prompts done → SESSION_READY")

            return self._make_response(
                pcm=np.zeros((1, 1, 1920), dtype=np.float32),
                text_token=np.array([0], dtype=np.int32),
                ready=np.array([True]),
            )

        # --- Regular frame: encode → LM step → decode (all in-process) ---
        t0 = time.monotonic()

        pcm_np = pb_utils.get_input_tensor_by_name(request, "INPUT_PCM").as_numpy()
        pcm = torch.from_numpy(pcm_np).to(device=_DEVICE, dtype=self._input_dtype)

        # Encode
        codes = self.mimi.encode(pcm)  # [1, 8, T]
        t1 = time.monotonic()

        # LM step + decode (same loop as server.py)
        pcm_frames = []
        last_text_token = 0

        for c in range(codes.shape[-1]):
            tokens = self.lm_gen.step(codes[:, :, c:c + 1])
            if tokens is None:
                continue
            last_text_token = int(tokens[0, 0, 0].item())
            main_pcm = self.mimi.decode(tokens[:, 1:9])  # [1, 1, 1920]
            pcm_frames.append(main_pcm)

        # Decode text token to string (incremental)
        new_text = ""
        if last_text_token > 0 and last_text_token < 32000:
            self._text_tokens.append(last_text_token)
            decoded = self.text_tokenizer.DecodeIds(self._text_tokens)
            new_text = decoded[self._last_decoded_len:]
            self._last_decoded_len = len(decoded)

        t2 = time.monotonic()

        if pcm_frames:
            out_pcm = torch.cat(pcm_frames, dim=-1).float().cpu().numpy()
        else:
            out_pcm = np.zeros((1, 1, 1920), dtype=np.float32)

        # CPU DSP resample 24kHz → 48kHz (polyphase filtering)
        # Smooths frame boundary transitions without hallucinating frequencies
        pcm_1d = out_pcm.reshape(-1).astype(np.float32)
        if _RESAMPLE == "soxr":
            pcm_48k = soxr.resample(pcm_1d, 24000, 48000, quality='VHQ')
        elif _RESAMPLE == "scipy":
            pcm_48k = resample_poly(pcm_1d, up=2, down=1).astype(np.float32)
        else:
            # Linear interpolation fallback
            n_out = len(pcm_1d) * 2
            pcm_48k = np.interp(
                np.linspace(0, len(pcm_1d) - 1, n_out),
                np.arange(len(pcm_1d)), pcm_1d
            ).astype(np.float32)
        out_48k = pcm_48k.reshape(1, 1, -1)
        t3 = time.monotonic()

        # Logging
        self._frame_count += 1
        if self._frame_count <= 5 or self._frame_count % 25 == 0:
            self.logger.log_info(
                f"frame {self._frame_count}: "
                f"enc={t1-t0:.3f}s lm+dec={t2-t1:.3f}s resample={t3-t2:.3f}s "
                f"total={t3-t0:.3f}s ({_RESAMPLE})"
            )

        return self._make_response(
            pcm=out_48k,
            text_token=np.array([last_text_token], dtype=np.int32),
            text_decoded=new_text,
            ready=np.array([False]),
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _make_response(pcm, text_token, ready, text_decoded=""):
        # Encode text_decoded as a numpy string array for Triton
        td_np = np.array([text_decoded], dtype=object)
        return pb_utils.InferenceResponse(output_tensors=[
            pb_utils.Tensor("OUTPUT_PCM_48K", pcm),
            pb_utils.Tensor("TEXT_TOKEN",     text_token),
            pb_utils.Tensor("TEXT_DECODED",   td_np),
            pb_utils.Tensor("SESSION_READY",  ready),
        ])

    def finalize(self):
        del self.lm_gen
        del self.mimi
        torch.cuda.empty_cache()
        self.logger.log_info("personaplex_pipeline: finalized.")
