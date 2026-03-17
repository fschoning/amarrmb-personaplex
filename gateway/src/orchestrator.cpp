// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: MIT
//
// gateway/src/orchestrator.cpp — Ping-Pong Orchestrator implementation

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
    if (energy < threshold_) {
        ++silent_count_;
    } else {
        silent_count_ = 0;
    }
    return silent_count_ >= hold_frames_;
}

void SilenceDetector::reset() { silent_count_ = 0; }

// ============================================================================
// TranscriptBuffer
// ============================================================================

TranscriptBuffer::TranscriptBuffer(int max_tokens)
    : max_tokens_(max_tokens)
{}

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
    // Keep decoded text from growing unbounded (~50K chars max)
    if (decoded_text_.size() > 50000) {
        decoded_text_ = decoded_text_.substr(decoded_text_.size() - 40000);
    }
}

void TranscriptBuffer::clear() {
    std::lock_guard<std::mutex> lk(mtx_);
    tokens_.clear();
    decoded_text_.clear();
}

std::vector<int32_t> TranscriptBuffer::recent_tokens(int n) const {
    std::lock_guard<std::mutex> lk(mtx_);
    int start = std::max(0, static_cast<int>(tokens_.size()) - n);
    return std::vector<int32_t>(tokens_.begin() + start, tokens_.end());
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

Orchestrator::Orchestrator(const Config& cfg,
                           std::shared_ptr<Session> sess,
                           SendFn send)
    : cfg_(cfg)
    , sess_(sess)
    , send_(send)
    , corrid_hot_(sess->corrid)
    , corrid_standby_(sess->corrid + 1000)
    , silence_(cfg.silence_threshold, cfg.silence_hold_frames)
{
    // Persona from session config
    persona_ = sess_->config.persona_prompt.empty()
        ? "You are a friendly and knowledgeable AI assistant."
        : sess_->config.persona_prompt;

    // Try to connect to brain
    try {
        brain_ = std::make_unique<BrainClient>(cfg_.brain_grpc_url, "brain");
        brain_available_ = brain_->is_ready();
        if (brain_available_)
            fprintf(stderr, "[orchestrator] Brain LLM available.\n");
        else
            fprintf(stderr, "[orchestrator] Brain LLM not ready — running without context switching.\n");
    } catch (const std::exception& e) {
        fprintf(stderr, "[orchestrator] Brain LLM unavailable: %s\n", e.what());
        brain_available_ = false;
    }
}

Orchestrator::~Orchestrator() {
    if (brain_thread_.joinable()) brain_thread_.join();
}

// ---------------------------------------------------------------------------
// run() — main loop, called from session_worker thread
// ---------------------------------------------------------------------------
void Orchestrator::run() {
    init_hot();

    std::string resp_id = "resp_" + sess_->session_id;
    std::string item_id = "item_" + sess_->session_id;

    std::vector<float> frame_buf(FRAME_SAMPLES_24K);

    while (!sess_->should_close.load()) {
        // ── Pop 80ms of PCM from ring buffer ──────────────────────────────
        size_t got = sess_->audio_in.pop(frame_buf.data(), FRAME_SAMPLES_24K, 200);
        if (got == 0) continue;

        // ── Send frame to hot node ─────────────────────────────────────────
        FrameOutput out;
        bool ok = false;
        try {
            ok = ts_hot_->send_frame(frame_buf.data(), FRAME_SAMPLES_24K, out);
        } catch (const std::exception& e) {
            fprintf(stderr, "[orchestrator] hot node error: %s\n", e.what());
            break;
        }
        if (!ok) break;

        // ── Audio → client ─────────────────────────────────────────────────
        if (!out.pcm_48k.empty()) {
            std::string b64 = encode_audio_delta(out.pcm_48k.data(), out.pcm_48k.size());
            send_(make_audio_delta(resp_id, item_id, b64));
        }

        // ── Text token → transcript ────────────────────────────────────────
        if (out.text_token > 0 && out.text_token < 32000) {
            transcript_.push_token(out.text_token);
        }
        // Decoded text from PP → transcript
        if (!out.text_decoded.empty()) {
            transcript_.push_text(out.text_decoded);
        }

        ++frame_no_;
        ++frames_since_trigger_;

        // Fire brain every ~5 seconds when not in-flight and there's new text
        constexpr int kBrainEveryFrames = 62;  // 62 × 80ms ≈ 5s
        if (brain_available_
            && !brain_in_flight_.load()
            && transcript_.size() >= 10
            && frames_since_trigger_ >= kBrainEveryFrames)
        {
            std::string prompt = build_brain_prompt(persona_);
            request_brain_async(prompt);
            frames_since_trigger_ = 0;  // reset for next brain cycle
        }

        // ── Switch every 30s ───────────────────────────────────────────────
        // At the 375-frame mark, if a boot payload is ready, wait for
        // silence then init standby + switch.  With MPS, the standby init
        // (~5s system prompts) runs on the same PP partition but brain is
        // on its own partition and won't cause contention.
        constexpr int kSwitchEveryFrames = 375;  // 375 × 80ms = 30s

        if (frames_since_trigger_ >= kSwitchEveryFrames
            && boot_ready_.load()
            && !switch_queued_.load())
        {
            bool silent = silence_.push_frame(out.pcm_48k);
            if (silent) {
                std::lock_guard<std::mutex> lk(boot_mtx_);
                if (!pending_boot_payload_.empty()) {
                    fprintf(stderr, "[orchestrator] 30s switch: silence detected, "
                                    "init standby + switch (frame %d).\n", frame_no_);
                    init_standby(pending_boot_payload_);
                    pending_boot_payload_.clear();
                    boot_ready_.store(false);

                    if (state_.load() == OrchestratorState::STANDBY_READY) {
                        execute_switch();
                    }
                }
            }
        } else {
            // Keep silence detector running even when not switching
            silence_.push_frame(out.pcm_48k);
        }
    }

    // ── Clean shutdown ────────────────────────────────────────────────────
    if (brain_thread_.joinable()) brain_thread_.join();
    try { if (ts_hot_)     ts_hot_->send_end();     } catch (...) {}
    try { if (ts_standby_) ts_standby_->send_end(); } catch (...) {}
    send_(make_audio_done(resp_id));
}

// ---------------------------------------------------------------------------
// init_hot — start the hot TritonSession (blocks ~2-5s for system prompts)
// ---------------------------------------------------------------------------
void Orchestrator::init_hot() {
    std::vector<uint8_t> voice_bytes;
    if (!sess_->config.voice_prompt_embedding.empty()) {
        const auto& emb = sess_->config.voice_prompt_embedding;
        voice_bytes = base64_decode(emb.data(), emb.size());
    }

    std::vector<int32_t> text_tokens = sess_->config.text_prompt_tokens;

    fprintf(stderr, "[orchestrator] init_hot: corrid=%ld voice_bytes=%zu text_tokens=%zu\n",
            corrid_hot_, voice_bytes.size(), text_tokens.size());

    ts_hot_ = std::make_unique<TritonSession>(
        cfg_.triton_url, cfg_.pipeline_model, corrid_hot_, cfg_.model_version);

    bool ok = ts_hot_->send_start(voice_bytes, text_tokens);
    if (!ok) throw std::runtime_error("hot node send_start failed");

    sess_->triton_ready.store(true);
    send_(make_session_ready(sess_->session_id));
    state_.store(OrchestratorState::HOT_ONLY);
    fprintf(stderr, "[orchestrator] hot node READY.\n");
}

// ---------------------------------------------------------------------------
// request_brain_async — fire off brain query on a background thread
// ---------------------------------------------------------------------------
void Orchestrator::request_brain_async(const std::string& prompt) {
    if (brain_in_flight_.load()) return;
    brain_in_flight_.store(true);
    state_.store(OrchestratorState::PRIMING);

    // Detach previous thread if done
    if (brain_thread_.joinable()) brain_thread_.join();

    brain_thread_ = std::thread([this, prompt]() {
        fprintf(stderr, "[brain] querying (prompt_len=%zu)...\n", prompt.size());
        auto t0   = std::chrono::steady_clock::now();
        std::string response = brain_->query(prompt, 200);
        auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "[brain] response in %ldms: %.80s...\n", ms, response.c_str());

        if (!response.empty()) {
            std::string boot = build_boot_payload(response);
            {
                std::lock_guard<std::mutex> lk(boot_mtx_);
                pending_boot_payload_ = std::move(boot);
            }
            boot_ready_.store(true);
        } else {
            fprintf(stderr, "[brain] empty response.\n");
        }

        // Always return to HOT_ONLY so brain can fire again immediately
        state_.store(OrchestratorState::HOT_ONLY);
        brain_in_flight_.store(false);
    });
}

// ---------------------------------------------------------------------------
// init_standby — send START to the standby node with the boot payload
// ---------------------------------------------------------------------------
void Orchestrator::init_standby(const std::string& boot_payload) {
    fprintf(stderr, "[orchestrator] init_standby: corrid=%ld\n", corrid_standby_);

    // Tear down any existing standby
    if (ts_standby_) {
        try { ts_standby_->send_end(); } catch (...) {}
        ts_standby_.reset();
    }

    ts_standby_ = std::make_unique<TritonSession>(
        cfg_.triton_url, cfg_.pipeline_model, corrid_standby_, cfg_.model_version);

    // Reuse the SAME voice as the hot node (from session config)
    std::vector<uint8_t> voice_bytes;
    if (!sess_->config.voice_prompt_embedding.empty()) {
        const auto& emb = sess_->config.voice_prompt_embedding;
        voice_bytes = base64_decode(emb.data(), emb.size());
    }

    // Carry over voice name tokens (e.g. [-999, 78, 65, 84, 70, 48] = "NATF0")
    std::vector<int32_t> text_tokens = sess_->config.text_prompt_tokens;

    fprintf(stderr, "[orchestrator] standby using voice_bytes=%zu text_tokens=%zu\n",
            voice_bytes.size(), text_tokens.size());

    // TODO(phase4): inject boot_payload as text context into the pipeline
    // For now: boot_payload is logged but not delivered to the LM.
    fprintf(stderr, "[orchestrator] boot_payload (%zu bytes): %.100s...\n",
            boot_payload.size(), boot_payload.c_str());

    bool ok = ts_standby_->send_start(voice_bytes, text_tokens);
    if (!ok) {
        fprintf(stderr, "[orchestrator] standby send_start failed — aborting switch.\n");
        ts_standby_.reset();
        state_.store(OrchestratorState::HOT_ONLY);
        return;
    }

    state_.store(OrchestratorState::STANDBY_READY);
    fprintf(stderr, "[orchestrator] standby READY. Waiting for silence...\n");
}

// ---------------------------------------------------------------------------
// execute_switch — swap hot ↔ standby at a silence boundary
// ---------------------------------------------------------------------------
void Orchestrator::execute_switch() {
    switch_queued_.store(true);
    state_.store(OrchestratorState::SWITCHING);

    // Swap the two sessions
    std::swap(ts_hot_, ts_standby_);
    std::swap(corrid_hot_, corrid_standby_);

    fprintf(stderr, "[orchestrator] SWITCHED: new hot corrid=%ld\n", corrid_hot_);

    // Tear down the old hot (now in ts_standby_ slot)
    if (ts_standby_) {
        try { ts_standby_->send_end(); } catch (...) {}
        ts_standby_.reset();
    }

    // Clear transcript — new node starts fresh
    transcript_.clear();
    silence_.reset();
    frames_since_trigger_ = 0;

    state_.store(OrchestratorState::HOT_ONLY);
    switch_queued_.store(false);

    fprintf(stderr, "[orchestrator] switch complete.\n");
}

// ---------------------------------------------------------------------------
// should_trigger_brain — returns true when we should ask the brain for context
// ---------------------------------------------------------------------------
bool Orchestrator::should_trigger_brain() const {
    // Trigger every ~10 seconds of speech (125 frames × 80ms = 10s)
    // AND only if we have at least 20 text tokens (some speech happened)
    // Trigger every ~30 seconds of audio (375 frames × 80ms = 30s).
    // Brain generation takes ~6s → choppy for ~6s every 30s (20% of cycle).
    // At 10s it was 60% choppy — unacceptable.
    constexpr int kTriggerEveryFrames = 375;
    constexpr int kMinTokens          = 20;
    return frames_since_trigger_ >= kTriggerEveryFrames
        && transcript_.size() >= kMinTokens;
}

// ---------------------------------------------------------------------------
// build_brain_prompt — assemble the prompt to send to the brain LLM
// ---------------------------------------------------------------------------
std::string Orchestrator::build_brain_prompt(const std::string& persona) const {
    std::string transcript = transcript_.as_text();
    int n_tokens = transcript_.size();
    int duration_s = frame_no_ * 80 / 1000;

    std::ostringstream oss;
    oss << "You are the brain behind a voice AI agent. "
        << "Your job is to write a context handoff note so a new instance "
        << "of the AI can continue the conversation seamlessly.\n\n"
        << "AI Persona: " << persona << "\n"
        << "Session duration: " << duration_s << " seconds\n"
        << "Speech tokens generated: " << n_tokens << "\n\n";

    if (!transcript.empty()) {
        // Include the last ~4000 chars of transcript (most recent ~3 minutes)
        std::string recent = transcript;
        if (recent.size() > 4000) {
            recent = "..." + recent.substr(recent.size() - 4000);
        }
        oss << "=== TRANSCRIPT (AI speech) ===\n"
            << recent << "\n"
            << "=== END TRANSCRIPT ===\n\n";
    } else {
        oss << "(No transcript yet -- conversation just started)\n\n";
    }

    oss << "Based on the transcript above, write a BOOT_PAYLOAD for the "
        << "next AI instance. Structure your response EXACTLY as:\n"
        << "[SUMMARY] What was being discussed (key topics, names, facts)\n"
        << "[CONTEXT] What the AI should know to continue naturally\n"
        << "[LAST_TOPIC] The most recent topic/question being addressed\n"
        << "[EMOTION] Tone of voice to match\n"
        << "[PERSONA] " << persona;

    return oss.str();
}

// ---------------------------------------------------------------------------
// build_boot_payload -- wrap brain response in PersonaPlex system format
// ---------------------------------------------------------------------------
std::string Orchestrator::build_boot_payload(const std::string& brain_response) const {
    std::ostringstream oss;
    oss << "<system>\n"
        << brain_response << "\n"
        << "</system>";
    return oss.str();
}

} // namespace pg
