// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: MIT
//
// gateway/src/brain_client.h — HTTP client for the brain FastAPI server
//
// Usage:
//   BrainClient brain(cfg.brain_grpc_url, "brain");
//   std::string boot_payload = brain.query(prompt, 64);

#pragma once

#include <cstdint>
#include <string>

namespace pg {

class BrainClient {
public:
    // url: "localhost:8015" or "brain:8015" — HTTP endpoint
    BrainClient(const std::string& url, const std::string& model_name = "brain");
    ~BrainClient() = default;

    // Send prompt to brain, return text response.
    // Returns empty string on error.
    std::string query(const std::string& prompt, int32_t max_tokens = 64);

    // Check if the brain server is ready.
    bool is_ready() const;

private:
    std::string base_url_;   // e.g., "http://localhost:8015"
    std::string model_name_;

    // Simple HTTP POST via libcurl
    static std::string http_post(const std::string& url,
                                 const std::string& json_body,
                                 long timeout_s = 120);
    static std::string http_get(const std::string& url, long timeout_s = 5);
};

} // namespace pg
