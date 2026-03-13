// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: MIT
//
// gateway/src/config.h — Runtime configuration (env vars + CLI flags)

#pragma once
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector> // Added for std::vector

namespace pg {

// Session-specific configuration, sent by client at session start
struct SessionConfig {
    std::string              instructions;           // text system prompt
    std::string              voice_prompt_embedding; // base64-encoded .pt bytes
    std::vector<int32_t>     text_prompt_tokens;     // optional: voice name as int32 sentinel
    std::string              persona_prompt;         // orchestrator persona (for brain LLM)
    std::string              input_audio_format  = "pcm16";
    std::string              output_audio_format = "pcm16";
    float                    temperature         = 0.8f;
    int                      top_k               = 250;
};

struct Config {
    // WebSocket server
    std::string ws_host        = "0.0.0.0";
    uint16_t    ws_port        = 8998;
    std::string ssl_cert       = "";          // empty = no TLS
    std::string ssl_key        = "";

    // Triton gRPC endpoint (localhost on same machine)
    std::string triton_url     = "localhost:8001";

    // Concurrency cap — must match instance_group.count in Triton configs
    int         max_sessions   = 6;

    // Session lifecycle
    int         session_timeout_s = 300;      // idle timeout before forceful close
    int         audio_buffer_ms   = 2000;     // ring buffer depth (ms at 24kHz)

    // Silence detection for Ping-Pong switch
    float       silence_threshold   = 0.005f;    // RMS below this = silent
    int         silence_hold_frames = 4;         // 4 × 80ms = 320ms silence window
    std::string pipeline_model = "personaplex_pipeline";
    int64_t     model_version  = -1;          // -1 = latest

    static Config from_env() {
        Config c;
        auto get = [](const char* var, const char* def) -> std::string {
            const char* v = std::getenv(var);
            return v ? v : def;
        };
        auto geti = [](const char* var, int def) -> int {
            const char* v = std::getenv(var);
            return v ? std::atoi(v) : def;
        };

        c.ws_host        = get("WS_HOST",        c.ws_host.c_str());
        c.ws_port        = static_cast<uint16_t>(geti("WS_PORT", c.ws_port));
        c.ssl_cert       = get("SSL_CERT",        "");
        c.ssl_key        = get("SSL_KEY",         "");
        c.triton_url     = get("TRITON_GRPC_URL", c.triton_url.c_str());
        c.max_sessions   = geti("MAX_SESSIONS",   c.max_sessions);
        c.session_timeout_s = geti("SESSION_TIMEOUT_S", c.session_timeout_s);
        c.pipeline_model = get("PIPELINE_MODEL",  c.pipeline_model.c_str());
        return c;
    }
};

} // namespace pg
