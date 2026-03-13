// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: MIT
//
// gateway/src/brain_client.h — Synchronous gRPC client for the "brain" Triton model
//
// Usage:
//   BrainClient brain(cfg.triton_url, "brain");
//   std::string boot_payload = brain.query(transcript, persona, max_tokens=256);

#pragma once

#include <cstdint>
#include <string>

// Triton gRPC client
#include <grpc_client.h>

namespace tc = triton::client;

namespace pg {

class BrainClient {
public:
    BrainClient(const std::string& triton_url, const std::string& model_name = "brain");
    ~BrainClient() = default;

    // Send prompt to brain, return text response.
    // Returns empty string on error.
    std::string query(const std::string& prompt, int32_t max_tokens = 256);

    // Check if the brain model is ready on Triton.
    bool is_ready() const;

private:
    std::unique_ptr<tc::InferenceServerGrpcClient> client_;
    std::string model_name_;
};

} // namespace pg
