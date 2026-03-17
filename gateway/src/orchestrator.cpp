// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: MIT
//
// gateway/src/orchestrator.cpp — PersonaPlex v3 Client-Driven Orchestrator

#include "orchestrator.h"
#include "audio_utils.h"
#include "protocol.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <sstream>

// Minimal JSON serialisation (no third-party dep, only used for simple events)
// For parsing we rely on protocol.cpp's parser.
static std::string json_str(const std::string& s) {
    std::string r = "\"";
    for (char c : s) {
        if (c == '"') r += "\\\"";
        else if (c == '\\') r += "\\\\";
        else if (c == '\n') r += "\\n";
        else r += c;
    }
    r += "\"";
    return r;
}

namespace pg {

using namespace pg::audio;
using namespace pg::proto;

// ============================================================================
// SilenceDetector
// ============================================================================

SilenceDetector::SilenceDetector(float threshold, int hold_frames)
    : threshold_(threshold), hold_frames_(hold_frames)
{}

float SilenceDetector::rms(const std::vector<float>& pcm) {
    if (pcm.empty()) return 0.0f;
    float sum = 0.0f;
    for (float s : pcm) sum += s * s;
    return std::sqrt(sum / static_cast<float>(pcm.size()));
}

bool SilenceDetector::push_frame(const std::vector<float>& pcm_48k) {
    float energy = rms(pcm_48k);
    if (energy < threshold_) { ++silent_count_; } else { silent_count_ = 0; }
    return silent_count_ >= hold_frames_;
}

void SilenceDetector::reset() { silent_count_ = 0; }

// ============================================================================
// TranscriptBuffer
// ============================================================================

TranscriptBuffer::TranscriptBuffer(int max_tokens) : max_tokens_(max_tokens) {}

void TranscriptBuffer::push_token(int32_t token) {
    if (token <= 0) return;
    std::lock_guard<std::mutex> lk(mtx_);
    tokens_.push_back(token);
    while (static_cast<int>(tokens_.size()) > max_tokens_)
        tokens_.pop_front();
}

void TranscriptBuffer::push_text(const std::string& text) {
    if (text.empty()) return;
    std::lock_guard<std::mutex> lk(mtx_);
    decoded_text_ += text;
    if (decoded_text_.size() > 50000)
        decoded_text_ = decoded_text_.substr(decoded_text_.size() - 40000);
}

void TranscriptBuffer::clear() {
    std::lock_guard<std::mutex> lk(mtx_);
    tokens_.clear();
    decoded_text_.clear();
}

std::vector<int32_t> TranscriptBuffer::recent_tokens(int n) const {
    std::lock_guard<std::mutex> lk(mtx_);
    int start = std::max(0, static_cast<int>(tokens_.size()) - n);
    return {tokens_.begin() + start, tokens_.end()};
}

std::string TranscriptBuffer::as_text() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return decoded_text_;
}

int TranscriptBuffer::size() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return static_cast<int>(tokens_.size());
}

// ============================================================================
// Orchestrator
// ============================================================================

Orchestrator::Orchestrator(const Config& cfg, std::shared_ptr<Session> sess, SendFn send)
    : cfg_(cfg), sess_(sess), send_(std::move(send))
{
    persona_ = sess_->config.instructions;
    filler_prompt_ = sess_->config.filler_prompt.empty()
        ? cfg_.default_filler_prompt
        : sess_->config.filler_prompt;

    // Register command handler so main.cpp can route node.* commands here
    {
        std::lock_guard<std::mutex> lk(sess_->command_handler_mtx);
        sess_->command_handler = [this](std::string cmd) {
            handle_command(std::move(cmd));
        };
    }
}

Orchestrator::~Orchestrator() {}

// ---------------------------------------------------------------------------
// run — main audio loop (blocks until session ends)
// ---------------------------------------------------------------------------
void Orchestrator::run() {
    // Init Active PP
    init_active();

    // NOTE: Filler is NOT pre-initialised at startup.
    // It will be loaded when the client sends node.prime(filler) or node.switch(filler).
    // Pre-loading was found to cause GPU contention with the Active node.

    // ── Audio loop ─────────────────────────────────────────────────────────
    static const std::string resp_id = "resp-0";
    static const std::string item_id = "item-0";
    static const int kHumanSilenceEndFrames = 10; // 500ms at 20ms/frame

    while (!sess_->should_close.load()) {

        // ── Process pending commands from client ───────────────────────────
        {
            std::vector<Command> cmds;
            {
                std::lock_guard<std::mutex> lk(cmd_mtx_);
                std::swap(cmds, cmd_queue_);
            }
            for (auto& cmd : cmds) {
                if (cmd.type == "node.switch") {
                    if (cmd.target == "standby") cmd_switch(NodeRole::Standby);
                    else if (cmd.target == "filler") cmd_switch(NodeRole::Filler);
                } else if (cmd.type == "node.prime") {
                    NodeRole r = (cmd.target == "standby") ? NodeRole::Standby : NodeRole::Filler;
                    cmd_prime(r, cmd.prompt, cmd.voice);
                } else if (cmd.type == "node.refresh") {
                    cmd_refresh(cmd.prompt, cmd.voice);
                } else if (cmd.type == "node.activate") {
                    NodeRole r = (cmd.target == "standby") ? NodeRole::Standby
                               : (cmd.target == "filler")  ? NodeRole::Filler
                                                           : NodeRole::Active;
                    cmd_activate(r);
                } else if (cmd.type == "node.pause") {
                    NodeRole r = (cmd.target == "active") ? NodeRole::Active
                               : (cmd.target == "filler") ? NodeRole::Filler
                                                          : NodeRole::Standby;
                    cmd_pause(r);
                } else if (cmd.type == "node.stop") {
                    if (cmd.target == "filler") cmd_stop(NodeRole::Filler);
                } else if (cmd.type == "session.configure") {
                    if (!cmd.prompt.empty()) {
                        filler_prompt_ = cmd.prompt;
                        fprintf(stderr, "[orchestrator] filler_prompt updated.\n");
                    }
                    send_node_status();
                }
            }
        }

        // ── Read next audio chunk from ring buffer ─────────────────────────
        std::vector<float> pcm_24k(FRAME_SAMPLES_24K, 0.f);
        size_t got = sess_->audio_in.pop(pcm_24k.data(), FRAME_SAMPLES_24K, 50);
        if (got == 0) {
            if (sess_->should_close.load()) break;
            continue;
        }
        if (got < FRAME_SAMPLES_24K)
            std::fill(pcm_24k.begin() + got, pcm_24k.end(), 0.f);

        // ── Turn boundary detection on INPUT audio (human speaking) ───────────
        // We detect silence in the *input* (human mic) to know when human
        // has finished speaking. Use a simple RMS threshold on pcm_24k.
        {
            float sum = 0.f;
            for (float s : pcm_24k) sum += s * s;
            float rms_in = std::sqrt(sum / static_cast<float>(pcm_24k.size()));
            bool input_silent = rms_in < 0.008f;  // ~-42 dBFS threshold

            if (input_silent) {
                if (++human_silent_frames_ == kHumanSilenceEndFrames) {
                    // Human just stopped speaking (debounced)
                    send_turn_boundary("human_end");
                }
            } else {
                human_silent_frames_ = 0;
            }
        }

        // ── Infer on Active session ────────────────────────────────────────
        FrameOutput out;
        bool ok = false;
        if (ts_active_) {
            ok = ts_active_->send_frame(pcm_24k.data(), pcm_24k.size(), out);
        }

        if (!ok) {
            fprintf(stderr, "[orchestrator] Active PP inference failed, skipping frame.\n");
            continue;
        }

        // ── Send audio to client ──────────────────────────────────────────
        if (!out.pcm_48k.empty()) {
            std::string b64 = encode_audio_delta(out.pcm_48k.data(), out.pcm_48k.size());
            send_(make_audio_delta(resp_id, item_id, b64));
        }

        // ── Transcript accumulation and streaming ─────────────────────────
        if (out.text_token > 0 && out.text_token < 32000) {
            transcript_.push_token(out.text_token);
        }
        if (!out.text_decoded.empty()) {
            transcript_.push_text(out.text_decoded);
            send_transcript_delta(out.text_decoded);
        }

        // ── AI speaking boundary detection ────────────────────────────────────
        if (!out.pcm_48k.empty()) {
            float sum = 0.f;
            for (float s : out.pcm_48k) sum += s * s;
            bool is_loud = std::sqrt(sum / out.pcm_48k.size()) > 0.01f;
            if (is_loud && !ai_speaking_) {
                ai_speaking_ = true;
                send_turn_boundary("ai_start");
            } else if (!is_loud && ai_speaking_) {
                ai_speaking_ = false;
            }
        }

        ++frame_no_;

        // ── Silence detection for output ──────────────────────────────────────────
        if (state_.load() == OrchestratorState::STANDBY_READY) {
            bool silent = silence_.push_frame(out.pcm_48k);
            if (silent) {
                fprintf(stderr, "[orchestrator] Standby ready + silence → switching.\n");
                execute_switch_to_standby();
            }
        } else {
            silence_.push_frame(out.pcm_48k);
        }
    }

    // ── Shutdown ───────────────────────────────────────────────────────────
    try { if (ts_active_)  ts_active_->send_end();  } catch (...) {}
    try { if (ts_standby_) ts_standby_->send_end(); } catch (...) {}
    try { if (ts_filler_)  ts_filler_->send_end();  } catch (...) {}
    send_(make_audio_done(resp_id));
}

// ---------------------------------------------------------------------------
// handle_command — called from WebSocket thread, queues command for run()
// ---------------------------------------------------------------------------
void Orchestrator::handle_command(const std::string& json_cmd) {
    // Minimal JSON parsing — extract "type", "target", "prompt", "to", "voice"
    auto extract = [&](const std::string& key) -> std::string {
        std::string search = "\"" + key + "\"";
        auto pos = json_cmd.find(search);
        if (pos == std::string::npos) return {};
        pos = json_cmd.find(':', pos);
        if (pos == std::string::npos) return {};
        pos = json_cmd.find('"', pos);
        if (pos == std::string::npos) return {};
        auto end = json_cmd.find('"', pos + 1);
        if (end == std::string::npos) return {};
        return json_cmd.substr(pos + 1, end - pos - 1);
    };

    Command cmd;
    cmd.type   = extract("type");
    // "to" is alias for "target" in node.switch
    cmd.target = extract("target");
    if (cmd.target.empty()) cmd.target = extract("to");
    cmd.prompt = extract("prompt");
    cmd.voice  = extract("voice");

    if (cmd.type.empty()) {
        fprintf(stderr, "[orchestrator] Unrecognised command: %.80s\n", json_cmd.c_str());
        return;
    }

    fprintf(stderr, "[orchestrator] Command: type=%s target=%s\n",
            cmd.type.c_str(), cmd.target.c_str());

    std::lock_guard<std::mutex> lk(cmd_mtx_);
    cmd_queue_.push_back(std::move(cmd));
}

// ---------------------------------------------------------------------------
// init_active — start Active TritonSession
// ---------------------------------------------------------------------------
void Orchestrator::init_active() {
    std::vector<uint8_t> voice_bytes;
    if (!sess_->config.voice_prompt_embedding.empty()) {
        const auto& emb = sess_->config.voice_prompt_embedding;
        voice_bytes = base64_decode(emb.data(), emb.size());
    }
    std::vector<int32_t> text_tokens = sess_->config.text_prompt_tokens;

    fprintf(stderr, "[orchestrator] init_active: corrid=%ld voice=%zu text=%zu persona='%.60s'\n",
            kCorridActive, voice_bytes.size(), text_tokens.size(),
            sess_->config.instructions.c_str());

    ts_active_ = std::make_unique<TritonSession>(
        cfg_.triton_url, cfg_.pipeline_model, kCorridActive, cfg_.model_version);

    if (!ts_active_->send_start(voice_bytes, text_tokens, sess_->config.instructions))
        throw std::runtime_error("Active node send_start failed");

    sess_->triton_ready.store(true);
    send_(make_session_ready(sess_->session_id));
    state_.store(OrchestratorState::HOT_ONLY);
    send_node_status();
    fprintf(stderr, "[orchestrator] Active node READY.\n");
}

// ---------------------------------------------------------------------------
// init_filler — start Filler TritonSession with its prompt
// ---------------------------------------------------------------------------
void Orchestrator::init_filler(const std::string& prompt) {
    std::vector<uint8_t> voice_bytes;
    if (!sess_->config.voice_prompt_embedding.empty()) {
        const auto& emb = sess_->config.voice_prompt_embedding;
        voice_bytes = base64_decode(emb.data(), emb.size());
    }
    std::vector<int32_t> text_tokens = sess_->config.text_prompt_tokens;

    // Filler tokens: prefer filler_text_tokens (voice + "hold on" instruction),
    // fall back to plain text_prompt_tokens (voice only — no instruction).
    // The dual-sentinel format [-999, voiceASCII..., -998, tok1, tok2...] is
    // decoded by the pipeline model to load both voice and text conditioning.
    const std::vector<int32_t>& filler_tokens =
        !sess_->config.filler_text_tokens.empty()
            ? sess_->config.filler_text_tokens
            : sess_->config.text_prompt_tokens;

    fprintf(stderr, "[orchestrator] init_filler: corrid=%ld tokens=%zu prompt=%.60s...\n",
            kCorridFiller, filler_tokens.size(), prompt.c_str());

    ts_filler_ = std::make_unique<TritonSession>(
        cfg_.triton_url, cfg_.pipeline_model, kCorridFiller, cfg_.model_version);

    if (!ts_filler_->send_start(voice_bytes, filler_tokens, sess_->config.filler_prompt)) {
        fprintf(stderr, "[orchestrator] Filler send_start failed.\n");
        ts_filler_.reset();
        return;
    }

    fprintf(stderr, "[orchestrator] Filler node READY (idle).\n");
}

// ---------------------------------------------------------------------------
// reset_filler — tear down and re-prime filler after use
// ---------------------------------------------------------------------------
void Orchestrator::reset_filler() {
    fprintf(stderr, "[orchestrator] reset_filler: tearing down and re-priming.\n");
    if (ts_filler_) {
        try { ts_filler_->send_end(); } catch (...) {}
        ts_filler_.reset();
    }
    // Re-prime in background
    std::thread t([this]() {
        try { init_filler(filler_prompt_); } catch (...) {}
    });
    t.detach();
}

// ---------------------------------------------------------------------------
// cmd_switch — primary switch command from client
// ---------------------------------------------------------------------------
void Orchestrator::cmd_switch(NodeRole to) {
    if (to == NodeRole::Standby) {
        if (!ts_standby_) {
            fprintf(stderr, "[orchestrator] node.switch to standby: standby not primed.\n");
            send_(make_error("node_error", "Standby not primed"));
            return;
        }
        // Wait for silence then switch — mark state and let run() loop do it
        state_.store(OrchestratorState::STANDBY_READY);
        fprintf(stderr, "[orchestrator] node.switch: waiting for silence to switch to standby.\n");

    } else if (to == NodeRole::Filler) {
        execute_switch_to_filler();
    }
}

// ---------------------------------------------------------------------------
// cmd_prime — load new context into standby or filler
// ---------------------------------------------------------------------------
void Orchestrator::cmd_prime(NodeRole target, const std::string& prompt, const std::string& voice) {
    if (target != NodeRole::Standby) {
        fprintf(stderr, "[orchestrator] node.prime: only standby supported.\n");
        return;
    }

    fprintf(stderr, "[orchestrator] node.prime standby: prompt=%.60s...\n", prompt.c_str());

    // Tear down existing standby
    if (ts_standby_) {
        try { ts_standby_->send_end(); } catch (...) {}
        ts_standby_.reset();
    }

    std::vector<uint8_t> voice_bytes;
    if (!sess_->config.voice_prompt_embedding.empty()) {
        const auto& emb = sess_->config.voice_prompt_embedding;
        voice_bytes = base64_decode(emb.data(), emb.size());
    }
    // TODO(v3.1): override voice if 'voice' param set
    std::vector<int32_t> text_tokens = sess_->config.text_prompt_tokens;

    ts_standby_ = std::make_unique<TritonSession>(
        cfg_.triton_url, cfg_.pipeline_model, kCorridStandby, cfg_.model_version);

    // Run send_start in a background thread so the audio loop is not blocked.
    // When done, set state to STANDBY_READY and notify client.
    state_.store(OrchestratorState::PRIMING);
    if (priming_thread_.joinable()) priming_thread_.join();

    priming_thread_ = std::thread([this, voice_bytes = std::move(voice_bytes),
                                   text_tokens = std::move(text_tokens),
                                   persona = sess_->config.instructions]() mutable {
        auto* ts = ts_standby_.get();
        if (!ts) return;
        if (!ts->send_start(voice_bytes, text_tokens, persona)) {
            fprintf(stderr, "[orchestrator] Standby send_start failed.\n");
            ts_standby_.reset();
            send_(make_error("node_error", "Standby prime failed"));
            state_.store(OrchestratorState::HOT_ONLY);
            return;
        }
        // Standby is ready — update state and notify client
        state_.store(OrchestratorState::STANDBY_READY);
        send_("{\"type\":\"node.standby_ready\"}");
        send_node_status();
        fprintf(stderr, "[orchestrator] Standby READY.\n");
    });
    priming_thread_.detach();
}

// ---------------------------------------------------------------------------
// cmd_activate — manual activate (testing)
// ---------------------------------------------------------------------------
void Orchestrator::cmd_activate(NodeRole target) {
    if (target == NodeRole::Standby) {
        execute_switch_to_standby();
    } else if (target == NodeRole::Filler) {
        execute_switch_to_filler();
    }
}

// ---------------------------------------------------------------------------
// cmd_pause — manual pause (testing)
// ---------------------------------------------------------------------------
void Orchestrator::cmd_pause(NodeRole target) {
    fprintf(stderr, "[orchestrator] node.pause %s (testing mode - frame routing continues)\n",
            target == NodeRole::Active ? "active" : "filler");
    // In testing mode: pause is a no-op, the active PP keeps generating
    // Full pause support (stopping frame sending) is a future feature
}

// ---------------------------------------------------------------------------
// cmd_refresh — keepalive: prime standby with new payload
// ---------------------------------------------------------------------------
void Orchestrator::cmd_refresh(const std::string& prompt, const std::string& voice) {
    fprintf(stderr, "[orchestrator] node.refresh: priming standby (keepalive).\n");
    cmd_prime(NodeRole::Standby, prompt, voice);
}

// ---------------------------------------------------------------------------
// cmd_stop — stop a node (filler: reset and re-prime)
// ---------------------------------------------------------------------------
void Orchestrator::cmd_stop(NodeRole target) {
    if (target == NodeRole::Filler) {
        // Make sure we're not in filler mode
        if (state_.load() == OrchestratorState::FILLER_ACTIVE) {
            // Switch back to active first — need a standby or we stay on active
            fprintf(stderr, "[orchestrator] node.stop filler: returning to active.\n");
            // Re-route audio back to original active (now ts_standby_ holds old active)
            if (ts_standby_) {
                std::swap(ts_active_, ts_standby_);
                ts_active_->send_end();  // this was the filler
            }
            state_.store(OrchestratorState::HOT_ONLY);
        }
        reset_filler();
        send_node_status();
    }
}

// ---------------------------------------------------------------------------
// execute_switch_to_standby
// ---------------------------------------------------------------------------
void Orchestrator::execute_switch_to_standby() {
    state_.store(OrchestratorState::SWITCHING);

    // Old active → tear down. Standby → becomes active.
    std::swap(ts_active_, ts_standby_);
    try { if (ts_standby_) { ts_standby_->send_end(); ts_standby_.reset(); } } catch (...) {}

    transcript_.clear();
    silence_.reset();
    frame_no_ = 0;

    state_.store(OrchestratorState::HOT_ONLY);
    send_node_status();
    fprintf(stderr, "[orchestrator] Switched: standby → active.\n");
}

// ---------------------------------------------------------------------------
// execute_switch_to_filler
// ---------------------------------------------------------------------------
void Orchestrator::execute_switch_to_filler() {
    if (!ts_filler_) {
        fprintf(stderr, "[orchestrator] node.switch to filler: filler not ready.\n");
        send_(make_error("node_error", "Filler not ready"));
        return;
    }

    state_.store(OrchestratorState::FILLER_ACTIVE);

    // Park current active in standby slot (so we can restore later)
    // Note: old standby (if any) is discarded
    if (ts_standby_) {
        try { ts_standby_->send_end(); } catch (...) {}
    }
    ts_standby_ = std::move(ts_active_);
    ts_active_  = std::move(ts_filler_);

    // Filler is now active — it will respond to incoming audio
    send_node_status();
    fprintf(stderr, "[orchestrator] Switched: filler → active. Old active parked as standby.\n");
}

// ---------------------------------------------------------------------------
// send_node_status
// ---------------------------------------------------------------------------
void Orchestrator::send_node_status() {
    std::string state_str;
    switch (state_.load()) {
        case OrchestratorState::HOT_ONLY:      state_str = "hot_only"; break;
        case OrchestratorState::PRIMING:       state_str = "priming"; break;
        case OrchestratorState::STANDBY_READY: state_str = "standby_ready"; break;
        case OrchestratorState::FILLER_ACTIVE: state_str = "filler_active"; break;
        case OrchestratorState::SWITCHING:     state_str = "switching"; break;
    }
    std::ostringstream oss;
    oss << "{\"type\":\"node.status\","
        << "\"active\":"   << (ts_active_  ? "true" : "false") << ","
        << "\"standby\":"  << (ts_standby_ ? "true" : "false") << ","
        << "\"filler\":"   << (ts_filler_  ? "true" : "false") << ","
        << "\"state\":"    << json_str(state_str) << "}";
    send_(oss.str());
}

// ---------------------------------------------------------------------------
// send_transcript_delta — stream decoded text to client
// ---------------------------------------------------------------------------
void Orchestrator::send_transcript_delta(const std::string& text) {
    std::ostringstream oss;
    oss << "{\"type\":\"transcript.delta\",\"text\":" << json_str(text)
        << ",\"frame\":" << frame_no_ << "}";
    send_(oss.str());
}

// ---------------------------------------------------------------------------
// send_turn_boundary — emit turn.boundary event to client
// ---------------------------------------------------------------------------
void Orchestrator::send_turn_boundary(const std::string& speaker) {
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    std::ostringstream oss;
    oss << "{\"type\":\"turn.boundary\","
        << "\"speaker\":" << json_str(speaker) << ","
        << "\"frame\":" << frame_no_ << ","
        << "\"ts_ms\":" << now_ms << "}";
    send_(oss.str());
    fprintf(stderr, "[orchestrator] turn.boundary speaker=%s frame=%d\n",
            speaker.c_str(), frame_no_);
}

} // namespace pg
