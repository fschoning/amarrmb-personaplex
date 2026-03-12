# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT
#
# Triton Python Backend — LavaSR v2
# Upsamples 24kHz PCM [1, 1, 1920] → 48kHz PCM [1, 1, 3840].
# Stateless — one instance handles all sessions via dynamic batching.
#
# Uses ysharma3501/LavaSR from https://github.com/ysharma3501/LavaSR
# HuggingFace model: YatharthS/LavaSR

import json
import os

import numpy as np
import torch
import torchaudio

import triton_python_backend_utils as pb_utils

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_HF_REPO = os.environ.get("LAVASR_HF_REPO", "YatharthS/LavaSR")


class TritonPythonModel:

    def initialize(self, args: dict):
        self.logger = pb_utils.Logger
        self.logger.log_info("lavasr_v2: loading model...")

        try:
            from LavaSR.model import LavaEnhance2
            self.model = LavaEnhance2(_HF_REPO, _DEVICE.type)
            self._mode = "lavasr"
            self.logger.log_info(f"lavasr_v2: loaded from {_HF_REPO} (mode=lavasr)")
        except Exception as e:
            raise RuntimeError(
                f"Cannot load LavaSR: {e}\n"
                "Install with: pip install git+https://github.com/ysharma3501/LavaSR.git"
            ) from e

        self.logger.log_info("lavasr_v2: ready.")

    @torch.no_grad()
    def execute(self, requests):
        responses = []

        for request in requests:
            pcm_np = pb_utils.get_input_tensor_by_name(
                request, "PCM_24K"
            ).as_numpy()   # [1, 1, 1920] float32, 24kHz

            pcm = torch.from_numpy(pcm_np).to(_DEVICE)  # [1, 1, 1920]

            # Clean resample 24kHz → 48kHz (no LavaSR for now)
            # LavaSR degrades quality on 80ms frames because:
            #   1) 24→16kHz downsample is lossy
            #   2) BWE has no context across frame boundaries
            pcm_2d = pcm.squeeze(0)  # [1, 1920]
            pcm_48k = torchaudio.functional.resample(pcm_2d, 24000, 48000)  # [1, 3840]

            out_np = pcm_48k.float().cpu().numpy().reshape(1, 1, -1)  # [1, 1, 3840]

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("OUTPUT_PCM_48K", out_np)]
                )
            )

        return responses

    def finalize(self):
        del self.model
        torch.cuda.empty_cache()
