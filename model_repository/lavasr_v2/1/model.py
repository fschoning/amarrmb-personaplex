# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT
#
# Triton Python Backend — LavaSR v2
# Upsamples 24kHz PCM [1, 1, 1920] → 48kHz PCM [1, 1, 3840].
# Stateless — one instance handles all sessions via dynamic batching.
#
# To switch to TensorRT after running scripts/export_lavasr.py:
#   1. Place model.plan in model_repository/lavasr_v2/1/
#   2. Change backend in config.pbtxt to "tensorrt" and remove model.py

import json
import os

import numpy as np
import torch

import triton_python_backend_utils as pb_utils

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_HF_REPO = os.environ.get("LAVASR_HF_REPO", "declinator/lava-sr-v2")


class TritonPythonModel:

    def initialize(self, args: dict):
        self.logger = pb_utils.Logger
        self.logger.log_info("lavasr_v2: loading model from HuggingFace...")

        try:
            # Primary: load via the lavasr package if installed
            from lavasr import LavaSR  # type: ignore
            self.model = LavaSR.from_pretrained(_HF_REPO).to(_DEVICE).eval()
            self._mode = "lavasr"
        except ImportError:
            # Fallback: load as a generic HF model / torch hub
            self.logger.log_warn(
                "lavasr package not found; attempting torch.hub.load from "
                + _HF_REPO
            )
            try:
                self.model = torch.hub.load(
                    _HF_REPO, "lavasr_v2", pretrained=True
                ).to(_DEVICE).eval()
                self._mode = "torchhub"
            except Exception as e:
                raise RuntimeError(
                    f"Cannot load LavaSR v2: {e}\n"
                    "Install with: pip install git+https://github.com/declinator/lava-sr"
                ) from e

        # Compile for SM 12.1 (DGX Spark Blackwell)
        if hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        self.logger.log_info(f"lavasr_v2: loaded (mode={self._mode}).")

    @torch.no_grad()
    def execute(self, requests):
        responses = []

        for request in requests:
            pcm_np = pb_utils.get_input_tensor_by_name(
                request, "PCM_24K"
            ).as_numpy()   # [1, 1, 1920] float32, 24kHz

            pcm = torch.from_numpy(pcm_np).to(_DEVICE)  # [1, 1, 1920]

            # LavaSR v2 accepts a raw waveform tensor and returns 48kHz output.
            # Input sample rate is passed so the model can adapt.
            upsampled = self.model(pcm, input_sr=24000, output_sr=48000)
            # upsampled: [1, 1, 3840]

            out_np = upsampled.float().cpu().numpy()  # [1, 1, 3840]

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("OUTPUT_PCM_48K", out_np)]
                )
            )

        return responses

    def finalize(self):
        del self.model
        torch.cuda.empty_cache()
