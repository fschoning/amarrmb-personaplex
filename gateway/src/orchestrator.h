// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: MIT
//
// gateway/src/orchestrator.h — Ping-Pong Orchestrator
//
// Owns two TritonSessions (hot + standby) and the Brain LLM client.
// Core responsibilities:
//   1. Route PCM: user audio in → hot node → audio out to client
//   2. Accumulate text tokens from hot node → rolling transcript
//   3. Detect switch triggers (keyword OR context threshold)
//   4. Query brain async for boot payload
//   5. Re-init standby with boot payload
//   6. On silence: hard-switch hot ↔ standby
//   7. Tear down old hot, free its Triton sequence slot

#pragma once

#include "brain_client.h"
#include "config.h"
#include "session.h"
#include "triton_client.h"

#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace pg {

// ---------------------------------------------------------------------------
// SilenceDetector — tracks RMS energy over a sliding window of frames
// ---------------------------------------------------------------------------
class SilenceDetector {
public:
    // threshold: RMS below this is "silence"
    // hold_frames: must be silent for this many consecutive frames
    explicit SilenceDetector(float threshold = 0.005f, int hold_frames = 4);

    // Returns true if the last `hold_frames` frames were all silent.
    bool push_frame(const std::vector<float>& pcm_48k);

    void reset();

private:
    float   threshold_;
    int     hold_frames_;
    int     silent_count_ = 0;

    static float rms(const std::vector<float>& pcm);
};

// ---------------------------------------------------------------------------
// TranscriptBuffer — accumulates text tokens into a rolling text transcript
// ---------------------------------------------------------------------------
class TranscriptBuffer {
public:
    explicit TranscriptBuffer(int max_tokens = 500);

    void push_token(int32_t token);
    void push_text(const std::string& text);  // Append decoded text from PP
    void clear();

    // Returns raw token IDs for recent N tokens
    std::vector<int32_t> recent_tokens(int n = 50) const;

    // Returns accumulated decoded text from the PP pipeline.
    std::string as_text() const;

    int size() const;

private:
    int                  max_tokens_;
    mutable std::mutex   mtx_;
    std::deque<int32_t>  tokens_;
    std::string          decoded_text_;  // Accumulated decoded text
};

// ---------------------------------------------------------------------------
// OrchestratorState — which Ping-Pong state are we in
// ---------------------------------------------------------------------------
enum class OrchestratorState {
    SINGLE,      // Only hot node, no standby (fallback / startup)
    HOT_ONLY,    // Hot is running, standby not yet initialised
    PRIMING,     // Brain query in flight + standby initialising
    STANDBY_READY,  // Standby is init'd; waiting for silence to switch
    SWITCHING,   // Switch in progress (teardown old hot)
};

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------
class Orchestrator {
public:
    // Callback types used to communicate back to the gateway layer
    using SendFn  = std::function<void(std::string)>;  // post JSON to client

    Orchestrator(const Config& cfg, std::shared_ptr<Session> sess, SendFn send);
    ~Orchestrator();

    // Main entry point — call from the session worker thread.
    // Blocks until session ends.
    void run();

private:
    // ---------- core loop helpers ----------
    void init_hot();
    void request_brain_async(const std::string& transcript);
    void init_standby(const std::string& boot_payload);
    void execute_switch();

    // ---------- trigger logic ----------
    bool should_trigger_brain() const;
    std::string build_brain_prompt(const std::string& persona) const;
    std::string build_boot_payload(const std::string& brain_response) const;

    // ---------- state ----------
    const Config&                    cfg_;
    std::shared_ptr<Session>         sess_;
    SendFn                           send_;

    // Triton sessions — hot is always active, standby may be null
    std::unique_ptr<TritonSession>   ts_hot_;
    std::unique_ptr<TritonSession>   ts_standby_;
    int64_t                          corrid_hot_     = 1;
    int64_t                          corrid_standby_ = 2;

    // Brain LLM client (optional — null if LOAD_BRAIN=0)
    std::unique_ptr<BrainClient>     brain_;
    bool                             brain_available_ = false;

    // Transcript
    TranscriptBuffer                 transcript_;

    // Silence detector
    SilenceDetector                  silence_;

    // Orchestrator state machine
    std::atomic<OrchestratorState>   state_{OrchestratorState::HOT_ONLY};

    // Async brain thread
    std::thread                      brain_thread_;
    std::atomic<bool>                brain_in_flight_{false};
    std::string                      pending_boot_payload_;
    mutable std::mutex               boot_mtx_;
    std::atomic<bool>                boot_ready_{false};

    // Switch gate: only queue one switch at a time
    std::atomic<bool>                switch_queued_{false};

    // Frame counters
    int                              frame_no_      = 0;
    int                              frames_since_trigger_ = 0;

    // Persona string from session config
    std::string                      persona_;
};

} // namespace pg
