// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: MIT
//
// gateway/src/triton_client.h — Wrapper around the Triton gRPC C++ client
//
// Usage per session:
//   TritonSession ts(cfg.triton_url, cfg.pipeline_model, session.corrid);
//   ts.send_start(voice_bytes, text_tokens);   // blocks ~2-5s → system prompt
//   while (active) {
//       ts.send_frame(pcm_1920, out_pcm_3840, out_text_token);
//   }
//   ts.send_end();

#pragma once

#include "config.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Triton gRPC client
#include <grpc_client.h>   // from libb_grpcclient.so in Triton SDK container

namespace tc = triton::client;

namespace pg {

struct FrameOutput {
    std::vector<float> pcm_48k;           // 3840 float32 samples (80ms at 48kHz)
    int32_t            text_token = 0;
    std::string        text_decoded;       // Incrementally decoded text from SentencePiece
    bool               session_ready = false;
};

class TritonSession {
public:
    TritonSession(const std::string& triton_url,
                  const std::string& model_name,
                  int64_t            corrid,
                  int64_t            model_version = -1);
    ~TritonSession();

    // Blocking — runs system-prompt conditioning in Triton (~2-5s).
    // Returns false on Triton error.
    bool send_start(const std::vector<uint8_t>& voice_prompt_bytes,
                    const std::vector<int32_t>& text_prompt_tokens,
                    const std::string& persona_text = "");

    // Blocking — sends one 80ms frame (1920 float32 samples at 24kHz).
    // Fills out on success; returns false on Triton error.
    bool send_frame(const float* pcm_24k, size_t n_samples, FrameOutput& out);

    // Send END=1 to cleanly release the Triton sequence slot.
    bool send_end();

private:
    std::unique_ptr<tc::InferenceServerGrpcClient> client_;
    std::string model_name_;
    int64_t     corrid_;
    int64_t     model_version_;
    bool        stream_open_ = false;

    // Build the control input tensors (START, END, CORRID)
    tc::InferInput* make_control_tensor(const char* name, float value);
    tc::InferInput* make_corrid_tensor();

    // Core infer helper
    bool infer_one(
        tc::InferInput*                          pcm_in,    // [1,1,1920] f32 or nullptr on START/END
        bool                                     is_start,
        bool                                     is_end,
        const std::vector<uint8_t>*              voice_bytes,      // nullptr except on START
        const std::vector<int32_t>*              text_tokens,      // nullptr except on START
        const std::string*                       persona_text,
        FrameOutput&                             out);
};

} // namespace pg
