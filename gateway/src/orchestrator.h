// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: MIT
//
// gateway/src/orchestrator.h — PersonaPlex v3 Orchestrator
//
// v3 Architecture (client-driven):
//   - Spark is a GENERIC PP server. No brain logic here.
//   - Three PP roles: Active (speaking), Standby (preloaded), Filler (instant ack)
//   - Client (workstation) sends commands via WebSocket to control nodes
//   - Transcript text streamed to client in real-time
//   - Client decides when/how to switch nodes and what context to load
//
// Node corrids: Active=1, Standby=1001, Filler=2001

#pragma once

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
    explicit SilenceDetector(float threshold = 0.005f, int hold_frames = 4);
    bool push_frame(const std::vector<float>& pcm_48k);
    void reset();

private:
    float threshold_;
    int   hold_frames_;
    int   silent_count_ = 0;

    static float rms(const std::vector<float>& pcm);
};

// ---------------------------------------------------------------------------
// TranscriptBuffer — accumulates decoded text from PP
// ---------------------------------------------------------------------------
class TranscriptBuffer {
public:
    explicit TranscriptBuffer(int max_tokens = 500);

    void push_token(int32_t token);
    void push_text(const std::string& text);
    void clear();

    std::vector<int32_t> recent_tokens(int n = 50) const;
    std::string as_text() const;
    int size() const;

private:
    int                  max_tokens_;
    mutable std::mutex   mtx_;
    std::deque<int32_t>  tokens_;
    std::string          decoded_text_;
};

// ---------------------------------------------------------------------------
// NodeRole — which PP instance are we referring to
// ---------------------------------------------------------------------------
enum class NodeRole { Active, Standby, Filler };

// ---------------------------------------------------------------------------
// OrchestratorState — coarse FSM state
// ---------------------------------------------------------------------------
enum class OrchestratorState {
    HOT_ONLY,      // Only Active running
    PRIMING,       // Standby being loaded (send_start in progress)
    STANDBY_READY, // Standby primed, waiting for node.switch command
    FILLER_ACTIVE, // Filler is active (Active paused), brain working
    SWITCHING,     // Switch in progress
};

// ---------------------------------------------------------------------------
// Orchestrator — v3 client-driven
// ---------------------------------------------------------------------------
class Orchestrator {
public:
    using SendFn = std::function<void(std::string)>;

    Orchestrator(const Config& cfg, std::shared_ptr<Session> sess, SendFn send);
    ~Orchestrator();

    // Main loop — blocks until session ends.
    void run();

    // Called by the WebSocket handler on incoming client commands.
    // Thread-safe.
    void handle_command(const std::string& json_cmd);

private:
    // ── PP session management ──────────────────────────────────────────────
    void init_active();
    void init_filler(const std::string& filler_prompt);
    void reset_filler();   // Tear down filler corrid and re-prime with same prompt

    // ── Node switching (client-commanded) ─────────────────────────────────
    void cmd_switch(NodeRole to);          // Switch to standby or filler
    void cmd_prime(NodeRole target,
                   const std::string& prompt,
                   const std::string& voice);
    void cmd_activate(NodeRole target);    // Manual (testing)
    void cmd_pause(NodeRole target);       // Manual (testing)
    void cmd_stop(NodeRole target);        // Reset filler after use
    void cmd_refresh(const std::string& prompt, const std::string& voice); // Keepalive refresh

    // ── Helpers ───────────────────────────────────────────────────────────
    TritonSession* active_session();
    void execute_switch_to_standby();
    void execute_switch_to_filler();
    void send_node_status();
    void send_transcript_delta(const std::string& text);
    void send_turn_boundary(const std::string& speaker); // "human_end" | "ai_start"

    // ── State ─────────────────────────────────────────────────────────────
    const Config&               cfg_;
    std::shared_ptr<Session>    sess_;
    SendFn                      send_;

    // Three PP sessions — only Active is always non-null after init
    std::unique_ptr<TritonSession>  ts_active_;
    std::unique_ptr<TritonSession>  ts_standby_;
    std::unique_ptr<TritonSession>  ts_filler_;

    // Fixed corrids for each role
    static constexpr int64_t kCorridActive  = 1;
    static constexpr int64_t kCorridStandby = 1001;
    static constexpr int64_t kCorridFiller  = 2001;

    // Filler prompt — set by session.configure, reused on reset
    std::string filler_prompt_;

    // Transcript
    TranscriptBuffer transcript_;

    // Silence detector (used for auto switch-on-silence if standby is ready)
    SilenceDetector silence_;

    // Orchestrator FSM
    std::atomic<OrchestratorState> state_{OrchestratorState::HOT_ONLY};

    // Pending commands from client (processed in run() loop)
    struct Command { std::string type; std::string target; std::string prompt; std::string voice; };
    std::vector<Command>  cmd_queue_;
    std::mutex            cmd_mtx_;

    // Background thread for standby priming (non-blocking)
    std::thread priming_thread_;

    // Frame counter
    int frame_no_ = 0;

    // Persona
    std::string persona_;

    // Turn boundary tracking
    // ai_speaking_: true while Active PP is producing non-silence audio
    bool ai_speaking_       = false;
    // human_silence_frames_: how many consecutive silent input frames
    int  human_silent_frames_ = 0;
    // After N silent input frames we treat it as human turn end
    static constexpr int kHumanSilenceEndFrames = 5;  // 5 × 80ms = 400ms
};

} // namespace pg
