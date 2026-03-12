# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT
#
# Triton Python Backend — Mimi Encoder
# Wraps MimiModel.encode(): raw PCM float32 [1,1,1920] → audio codes int32 [1,8,1]
# One Triton instance handles one session (sequence batcher guarantees this).
# START=1 → reset streaming state; END=1 → cleanup; regular → encode frame.

import json
import os

import numpy as np
import torch

import triton_python_backend_utils as pb_utils

from huggingface_hub import hf_hub_download
from moshi.models import loaders

_HF_REPO = os.environ.get("HF_REPO", "") or loaders.DEFAULT_REPO
_DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_mimi_weight_path():
    """Download Mimi weights from HuggingFace if not cached."""
    return hf_hub_download(_HF_REPO, loaders.MIMI_NAME)


class TritonPythonModel:
    """One instance of this class is created per model instance (max 6)."""

    def initialize(self, args: dict):
        self.logger = pb_utils.Logger

        self.logger.log_info("mimi_encoder: loading Mimi weights...")
        mimi_weight = _get_mimi_weight_path()
        self.mimi = loaders.get_mimi(mimi_weight, _DEVICE)

        # FP16 + torch.compile mirrors the existing optimisation in server.py
        self.mimi = self.mimi.half()
        self.mimi.torch_compile_encoder_decoder = True
        self.mimi = torch.compile(self.mimi)
        self.logger.log_info("mimi_encoder: Mimi loaded and compiled.")

        # Put in streaming mode (stays for the lifetime of this instance).
        self.mimi.streaming_forever(1)
        self._active = False

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

            if start:
                # New session on this instance — reset streaming state.
                self.mimi.reset_streaming()
                self._active = True
                self.logger.log_info("mimi_encoder: stream started (START=1)")

            if end:
                # Session finished — reset so next session starts clean.
                self.mimi.reset_streaming()
                self._active = False
                # Return empty CODES so the ensemble can propagate END.
                empty = np.zeros((1, 8, 1), dtype=np.int32)
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[pb_utils.Tensor("CODES", empty)]
                    )
                )
                continue

            # --- normal frame ---
            pcm_np = pb_utils.get_input_tensor_by_name(request, "INPUT_PCM").as_numpy()
            # pcm_np: [1, 1, 1920] float32
            pcm = torch.from_numpy(pcm_np).to(_DEVICE).half()

            codes = self.mimi.encode(pcm)          # [1, 8, Tframe] — typically Tframe=1
            codes_np = codes.cpu().numpy().astype(np.int32)

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("CODES", codes_np)]
                )
            )

        return responses

    def finalize(self):
        del self.mimi
        torch.cuda.empty_cache()
