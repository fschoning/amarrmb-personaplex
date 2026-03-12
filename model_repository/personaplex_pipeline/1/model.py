# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT
#
# Triton BLS (Business Logic Scripting) model — PersonaPlex Pipeline
# Replaces the ensemble because Triton ensembles don't support sequence batching.
#
# Chains: mimi_encoder → personaplex_lm → mimi_decoder → lavasr_v2
# Supports sequence batching (START/END/CORRID managed by Triton).

import time
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args: dict):
        self.logger = pb_utils.Logger
        self._frame_count = 0
        self.logger.log_info("personaplex_pipeline BLS: initialized")

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                resp = self._process_one(request)
                responses.append(resp)
            except Exception as e:
                self.logger.log_error(f"personaplex_pipeline BLS error: {e}")
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(str(e))
                ))

        return responses

    def _process_one(self, request):
        # Extract inputs from the gateway request
        input_pcm = pb_utils.get_input_tensor_by_name(request, "INPUT_PCM")
        corrid_t  = pb_utils.get_input_tensor_by_name(request, "CORRID")
        start_t   = pb_utils.get_input_tensor_by_name(request, "START")
        end_t     = pb_utils.get_input_tensor_by_name(request, "END")
        vp_bytes  = pb_utils.get_input_tensor_by_name(request, "VOICE_PROMPT_BYTES")
        tp_tokens = pb_utils.get_input_tensor_by_name(request, "TEXT_PROMPT_TOKENS")

        # Determine sequence flags from the request
        flags = request.flags()

        t0 = time.monotonic()

        # --- Step 1: mimi_encoder ---
        enc_request = pb_utils.InferenceRequest(
            model_name="mimi_encoder",
            inputs=[input_pcm, corrid_t, start_t, end_t],
            requested_output_names=["CODES"],
            flags=flags,
            correlation_id=request.correlation_id(),
        )
        enc_response = enc_request.exec()
        if enc_response.has_error():
            raise RuntimeError(f"mimi_encoder: {enc_response.error().message()}")

        codes = pb_utils.get_output_tensor_by_name(enc_response, "CODES")
        t1 = time.monotonic()

        # --- Step 2: personaplex_lm ---
        lm_inputs = [codes, corrid_t, start_t, end_t]
        if vp_bytes is not None:
            lm_inputs.append(vp_bytes)
        if tp_tokens is not None:
            lm_inputs.append(tp_tokens)

        lm_request = pb_utils.InferenceRequest(
            model_name="personaplex_lm",
            inputs=lm_inputs,
            requested_output_names=["TOKENS", "TEXT_TOKEN", "SESSION_READY"],
            flags=flags,
            correlation_id=request.correlation_id(),
        )
        lm_response = lm_request.exec()
        if lm_response.has_error():
            raise RuntimeError(f"personaplex_lm: {lm_response.error().message()}")

        tokens       = pb_utils.get_output_tensor_by_name(lm_response, "TOKENS")
        text_token   = pb_utils.get_output_tensor_by_name(lm_response, "TEXT_TOKEN")
        session_rdy  = pb_utils.get_output_tensor_by_name(lm_response, "SESSION_READY")
        t2 = time.monotonic()

        # --- Step 3: mimi_decoder ---
        dec_request = pb_utils.InferenceRequest(
            model_name="mimi_decoder",
            inputs=[tokens, corrid_t, start_t, end_t],
            requested_output_names=["PCM_24K"],
            flags=flags,
            correlation_id=request.correlation_id(),
        )
        dec_response = dec_request.exec()
        if dec_response.has_error():
            raise RuntimeError(f"mimi_decoder: {dec_response.error().message()}")

        pcm_24k = pb_utils.get_output_tensor_by_name(dec_response, "PCM_24K")
        t3 = time.monotonic()

        # Output 24kHz PCM directly (no upsampling — saves an IPC round-trip)
        # Keep tensor name "OUTPUT_PCM_48K" for gateway compatibility
        output_pcm = pb_utils.Tensor("OUTPUT_PCM_48K", pcm_24k.as_numpy())

        self._frame_count += 1
        if self._frame_count <= 5 or self._frame_count % 25 == 0:
            self.logger.log_info(
                f"BLS frame {self._frame_count}: "
                f"enc={t1-t0:.3f}s lm={t2-t1:.3f}s dec={t3-t2:.3f}s "
                f"total={t3-t0:.3f}s"
            )

        # --- Build response ---
        return pb_utils.InferenceResponse(
            output_tensors=[output_pcm, text_token, session_rdy]
        )

    def finalize(self):
        self.logger.log_info("personaplex_pipeline BLS: finalized")
