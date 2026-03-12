# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT
#
# Triton Python Backend — Audio Upsampler
# Resamples 24kHz PCM [1, 1, 1920] → 48kHz PCM [1, 1, 3840].
# Uses high-quality polyphase filtering on CPU (no GPU needed).
# Stateless — one instance handles all sessions via dynamic batching.

import numpy as np
import triton_python_backend_utils as pb_utils

# Use scipy for high-quality polyphase resampling on CPU
try:
    from scipy.signal import resample_poly
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class TritonPythonModel:

    def initialize(self, args: dict):
        self.logger = pb_utils.Logger
        if _HAS_SCIPY:
            self.logger.log_info("lavasr_v2: using scipy polyphase resampler (CPU)")
        else:
            self.logger.log_info("lavasr_v2: using numpy linear resampler (CPU)")
        self.logger.log_info("lavasr_v2: ready (CPU-only, no GPU needed).")

    def execute(self, requests):
        responses = []

        for request in requests:
            pcm_np = pb_utils.get_input_tensor_by_name(
                request, "PCM_24K"
            ).as_numpy()   # [1, 1, 1920] float32, 24kHz

            # Squeeze to 1D for resampling
            pcm_1d = pcm_np.reshape(-1)  # [1920]

            # High-quality polyphase resample: 24kHz → 48kHz (ratio = 2/1)
            if _HAS_SCIPY:
                pcm_48k = resample_poly(pcm_1d, up=2, down=1).astype(np.float32)
            else:
                # Fallback: simple linear interpolation
                n_out = len(pcm_1d) * 2
                indices = np.linspace(0, len(pcm_1d) - 1, n_out)
                pcm_48k = np.interp(indices, np.arange(len(pcm_1d)), pcm_1d).astype(np.float32)

            out_np = pcm_48k.reshape(1, 1, -1)  # [1, 1, 3840]

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("OUTPUT_PCM_48K", out_np)]
                )
            )

        return responses

    def finalize(self):
        pass
