# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT
#
# Triton Python Backend — PersonaPlex Language Model
#
# Frame flow:
#   START=1  → reset_streaming, load voice prompt & text prompt,
#              run system-prompt conditioning (~2-5s), return SESSION_READY=True
#   regular  → LMGen.step(codes) → tokens + text_token, SESSION_READY=False
#   END=1    → cleanup, SESSION_READY=False
#
# One Triton instance handles one session (sequence batcher guarantees this).
# All 6 instances share a single Mimi model for system-prompt encoding
# (protected by a threading.Lock — only one system prompt runs at a time).

import asyncio
import json
import os
import tempfile
import threading
from typing import Optional

import numpy as np
import sentencepiece
import torch

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

# ---------------------------------------------------------------------------
# Shared Mimi instance for system-prompt encoding (one per Triton process)
# ---------------------------------------------------------------------------
_shared_mimi_lock: threading.Lock = threading.Lock()
_shared_mimi: Optional[object] = None


def _get_shared_mimi():
    global _shared_mimi
    with _shared_mimi_lock:
        if _shared_mimi is None:
            mimi_weight = hf_hub_download(_HF_REPO, loaders.MIMI_NAME)
            m = loaders.get_mimi(mimi_weight, _DEVICE)
            m = m.half()
            m.torch_compile_encoder_decoder = True
            m = torch.compile(m)
            m.streaming_forever(1)
            _shared_mimi = m
        return _shared_mimi


# ---------------------------------------------------------------------------

class TritonPythonModel:

    def initialize(self, args: dict):
        self.logger = pb_utils.Logger
        self.logger.log_info("personaplex_lm: loading LM weights (this takes a moment)...")

        lm_weight = hf_hub_download(_HF_REPO, loaders.MOSHI_NAME)
        lm = loaders.get_moshi_lm(lm_weight, device=_DEVICE)

        if _USE_FP8:
            self.logger.log_info("personaplex_lm: applying FP8 quantisation...")
            quantize_model(lm)

        # Load text tokenizer
        tokenizer_path = hf_hub_download(_HF_REPO, loaders.TEXT_TOKENIZER_NAME)
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

        # Create a streaming LMGen wrapper (mirrors server.py)
        self.lm_gen = LMGen(
            lm,
            device=_DEVICE,
            temp=_TEMPERATURE,
            top_k=_TOP_K,
        )
        self.lm_gen.streaming_forever(1)

        # Session state populated on each START
        self._voice_prompt_bytes: Optional[bytes] = None
        self._text_prompt_tokens: Optional[list]  = None
        self._active = False

        self.logger.log_info("personaplex_lm: ready.")

    # ------------------------------------------------------------------

    def _run_system_prompts(self):
        """Load voice/text prompts and run conditioning; called once per START."""
        mimi = _get_shared_mimi()

        with _shared_mimi_lock:
            mimi.reset_streaming()

            # Load voice prompt embeddings from the bytes the client uploaded.
            # load_voice_prompt_embeddings() expects a file path (calls torch.load(path)),
            # so we write the bytes to a temp file first.
            if self._voice_prompt_bytes:
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

            # Run the synchronous wrapper of step_system_prompts_async.
            # asyncio.run() is safe here — Triton execute() runs in a thread pool,
            # not in an event loop.
            asyncio.run(
                self.lm_gen.step_system_prompts_async(mimi, is_alive=lambda: True)
            )

    # ------------------------------------------------------------------

    @torch.no_grad()
    def execute(self, requests):
        responses = []

        for request in requests:
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
                self._active = False
                responses.append(self._make_response(
                    tokens=np.zeros((1, 9, 1), dtype=np.int32),
                    text_token=np.array([0], dtype=np.int32),
                    ready=np.array([False]),
                ))
                continue

            # --- START ---
            if start:
                self.lm_gen.reset_streaming()
                self._active = True

                # Extract optional voice prompt bytes
                vp_tensor = pb_utils.get_input_tensor_by_name(
                    request, "VOICE_PROMPT_BYTES"
                )
                if vp_tensor is not None:
                    raw = vp_tensor.as_numpy()
                    if raw.size > 0:
                        # TYPE_STRING comes as numpy object array of bytes items
                        self._voice_prompt_bytes = raw.flat[0]

                # Extract optional text prompt tokens
                tp_tensor = pb_utils.get_input_tensor_by_name(
                    request, "TEXT_PROMPT_TOKENS"
                )
                if tp_tensor is not None:
                    tok = tp_tensor.as_numpy()
                    self._text_prompt_tokens = tok.tolist() if tok.size > 0 else None

                # Run conditioning (blocking, ~2-5s). When this returns,
                # the gateway sends session.ready to the client.
                self.logger.log_info("personaplex_lm: running system prompts...")
                self._run_system_prompts()
                self.logger.log_info("personaplex_lm: system prompts done → SESSION_READY")

                responses.append(self._make_response(
                    tokens=np.zeros((1, 9, 1), dtype=np.int32),
                    text_token=np.array([0], dtype=np.int32),
                    ready=np.array([True]),
                ))
                continue

            # --- Regular frame ---
            codes_np = pb_utils.get_input_tensor_by_name(
                request, "CODES"
            ).as_numpy()   # [1, 8, Tframe]

            codes = torch.from_numpy(codes_np).to(_DEVICE)  # int32

            all_tokens: list[torch.Tensor] = []
            last_text_token = 0

            # Iterate over time steps (usually 1 per frame)
            for c in range(codes.shape[-1]):
                result = self.lm_gen.step(codes[:, :, c:c+1])
                if result is not None:
                    # result: [1, dep_q+1, 1]  — row 0 = text token, rows 1–8 = audio
                    text_tok = int(result[0, 0, 0].item())
                    last_text_token = text_tok
                    all_tokens.append(result)

            if all_tokens:
                tokens_out = torch.cat(all_tokens, dim=-1).cpu().numpy().astype(np.int32)
            else:
                tokens_out = np.zeros((1, 9, 1), dtype=np.int32)

            responses.append(self._make_response(
                tokens=tokens_out,
                text_token=np.array([last_text_token], dtype=np.int32),
                ready=np.array([False]),
            ))

        return responses

    # ------------------------------------------------------------------

    @staticmethod
    def _make_response(tokens, text_token, ready):
        return pb_utils.InferenceResponse(output_tensors=[
            pb_utils.Tensor("TOKENS",        tokens),
            pb_utils.Tensor("TEXT_TOKEN",    text_token),
            pb_utils.Tensor("SESSION_READY", ready),
        ])

    def finalize(self):
        del self.lm_gen
        torch.cuda.empty_cache()
