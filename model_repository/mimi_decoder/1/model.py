# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT
#
# Triton Python Backend — Mimi Decoder
# Wraps MimiModel.decode(): LM tokens int32 [1,8,1] → PCM float32 [1,1,1920]
# One Triton instance handles one session (sequence batcher guarantees this).

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

    def initialize(self, args: dict):
        self.logger = pb_utils.Logger
        self.logger.log_info("mimi_decoder: loading Mimi weights...")

        mimi_weight = _get_mimi_weight_path()
        self.mimi = loaders.get_mimi(mimi_weight, _DEVICE)
        self.mimi = self.mimi.half()
        self.mimi.torch_compile_encoder_decoder = True
        self.mimi = torch.compile(self.mimi)

        self.mimi.streaming_forever(1)

        # Per-instance pinned CPU buffer for DtoH transfer (avoids race between instances)
        self._pinned_pcm = torch.empty(1920, dtype=torch.float32, pin_memory=True)
        self.logger.log_info("mimi_decoder: ready.")

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
                self.mimi.reset_streaming()
                self.logger.log_info("mimi_decoder: stream started")

            if end:
                self.mimi.reset_streaming()
                silence = np.zeros((1, 1, 1920), dtype=np.float32)
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[pb_utils.Tensor("PCM_24K", silence)]
                    )
                )
                continue

            tokens_np = pb_utils.get_input_tensor_by_name(request, "TOKENS").as_numpy()
            # tokens_np: [1, 9, Tframe]  — dep_q+1 = 9 rows; row 0 = text token
            # Mimi decoder expects rows 1–8 (the 8 audio codebook tokens)
            tokens = torch.from_numpy(tokens_np).to(_DEVICE)   # int32

            # Iterate over time steps (usually 1 per 80ms frame)
            pcm_frames = []
            for c in range(tokens.shape[-1]):
                audio_tokens = tokens[:, 1:9, c:c+1]           # [1, 8, 1]
                pcm_frame = self.mimi.decode(audio_tokens)      # [1, 1, 1920]
                pcm_frames.append(pcm_frame)

            if pcm_frames:
                pcm = torch.cat(pcm_frames, dim=-1)             # [1, 1, N*1920]
            else:
                pcm = torch.zeros(1, 1, 1920, device=_DEVICE)

            # Pinned DtoH (saves ~0.7ms vs .cpu())
            pcm_f = pcm.float()[0, 0]                           # [1920]
            self._pinned_pcm.copy_(pcm_f, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            pcm_np = self._pinned_pcm.numpy().reshape(1, 1, 1920)

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("PCM_24K", pcm_np)]
                )
            )

        return responses

    def finalize(self):
        del self.mimi
        torch.cuda.empty_cache()
