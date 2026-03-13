// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: MIT
//
// gateway/src/protocol.h — OpenAI Realtime API event types

#pragma once
#include "config.h"    // SessionConfig lives here
#include <string>
#include <vector>
#include <optional>

namespace pg::proto {

// ---------------------------------------------------------------------------
// Client → Server events (subset of OpenAI Realtime API)
// ---------------------------------------------------------------------------

enum class ClientEventType {
    Unknown,
    SessionUpdate,          // session.update
    InputAudioBufferAppend, // input_audio_buffer.append
    InputAudioBufferCommit, // input_audio_buffer.commit
    InputAudioBufferClear,  // input_audio_buffer.clear
    ResponseCreate,         // response.create
    ResponseCancel,         // response.cancel
};

// SessionConfig is defined in config.h


struct ClientEvent {
    ClientEventType         type      = ClientEventType::Unknown;
    std::string             event_id;
    // session.update
    SessionConfig           session;
    // input_audio_buffer.append
    std::string             audio_b64;  // base64 PCM16 24kHz
    // response.create / response.cancel
    std::string             response_id;
};

// ---------------------------------------------------------------------------
// Server → Client events
// ---------------------------------------------------------------------------

/// Serialise session.created / session.updated
std::string make_session_created(const std::string& session_id,
                                 const SessionConfig& cfg);
std::string make_session_updated(const std::string& session_id,
                                 const SessionConfig& cfg);

/// session.ready — our extension: signals system prompts are done
std::string make_session_ready(const std::string& session_id);

/// response.audio.delta
std::string make_audio_delta(const std::string& response_id,
                             const std::string& item_id,
                             const std::string& audio_b64);

/// response.audio.done
std::string make_audio_done(const std::string& response_id);

/// response.text.delta
std::string make_text_delta(const std::string& response_id,
                            const std::string& delta);

/// input_audio_buffer.speech_started / speech_stopped
std::string make_speech_started(int64_t audio_start_ms);
std::string make_speech_stopped(int64_t audio_end_ms);

/// error
std::string make_error(const std::string& type,
                       const std::string& message,
                       const std::string& event_id = "");

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Parse a raw JSON message string into a ClientEvent.
/// Returns false if parsing fails (sends an error event).
bool parse_client_event(const std::string& json_str, ClientEvent& out);
bool parse_client_event(const char* data, size_t len, ClientEvent& out);

} // namespace pg::proto
