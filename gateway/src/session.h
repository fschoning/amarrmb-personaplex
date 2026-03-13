// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: MIT
//
// gateway/src/session.h — Per-connection session state and SessionManager

#pragma once

#include "config.h"
#include "protocol.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// Forward-declare uWebSockets types to avoid pulling the full header here
namespace uWS { template<bool, bool, typename> struct WebSocket; }

namespace pg {

// ---------------------------------------------------------------------------
// Thread-safe ring buffer for 24kHz PCM float32 samples
// ---------------------------------------------------------------------------
class AudioRingBuffer {
public:
    explicit AudioRingBuffer(size_t capacity);
    // Push samples; returns number actually pushed (may be less if near-full)
    size_t push(const float* samples, size_t n);
    // Block until n samples available or timeout_ms expires; returns 0 on timeout
    size_t pop(float* out, size_t n, int timeout_ms = 200);
    void   reset();
    size_t available() const;

private:
    std::vector<float>      buf_;
    size_t                  capacity_;
    size_t                  head_ = 0, tail_ = 0, count_ = 0;
    mutable std::mutex      mtx_;
    std::condition_variable cv_;
};

// ---------------------------------------------------------------------------
// Session — one per connected WebSocket client
// ---------------------------------------------------------------------------
struct Session {
    // Unique identifier posted to Triton as CORRID
    int64_t            corrid;
    std::string        session_id;   // UUID string sent to client

    // Config populated from session.update
    SessionConfig       config;
    std::atomic<bool>    config_set{false};

    // Audio buffer: gateway writes, session thread reads
    AudioRingBuffer     audio_in;

    // Output queue: session thread writes, gateway uWS thread sends
    struct OutputMessage {
        std::string json;          // pre-serialised JSON event
    };
    std::queue<OutputMessage>  output_q;
    std::mutex                 output_mtx;

    // Lifecycle flags (atomic for cross-thread visibility)
    std::atomic<bool>  should_close{false};
    std::atomic<bool>  ready{false};         // true after session.update received
    std::atomic<bool>  triton_ready{false};  // true after SESSION_READY from LM

    // The worker thread that drives Triton inference for this session
    std::thread        worker;

    // Callback to post a message back onto the uWS event loop
    // Set once on session creation; safe to call from any thread.
    std::function<void(std::string)> send_to_client;

    explicit Session(int64_t id, std::string sid, int buf_ms);
    ~Session();

    Session(const Session&)            = delete;
    Session& operator=(const Session&) = delete;
};

// ---------------------------------------------------------------------------
// SessionManager — thread-safe map of ws pointer → Session
// ---------------------------------------------------------------------------
class SessionManager {
public:
    explicit SessionManager(const Config& cfg);

    // Called from uWS onOpen; returns nullptr if at capacity
    std::shared_ptr<Session> create(
        void* ws_key,
        std::function<void(std::string)> send_fn);

    // Called from uWS onMessage / onClose — O(1) lookup
    std::shared_ptr<Session> get(void* ws_key) const;

    // Called from uWS onClose — removes and signals the worker thread
    void remove(void* ws_key);

    int  active_count() const;
    int  max_sessions()  const { return cfg_.max_sessions; }

private:
    Config                                                     cfg_;
    std::unordered_map<void*, std::shared_ptr<Session>>        sessions_;
    mutable std::mutex                                         mtx_;
    std::atomic<int64_t>                                       next_corrid_{1};
};

} // namespace pg
