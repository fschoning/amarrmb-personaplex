// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: MIT
//
// gateway/src/protocol.cpp — OpenAI Realtime API serialisers and parser

#include "protocol.h"

#include <simdjson.h>   // header-only fast JSON parser

#include <sstream>
#include <string>

namespace pg::proto {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::string escape_json(const std::string& s) {
    std::ostringstream o;
    for (char c : s) {
        switch (c) {
            case '"':  o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\n': o << "\\n";  break;
            case '\r': o << "\\r";  break;
            case '\t': o << "\\t";  break;
            default:   o << c;
        }
    }
    return o.str();
}

static std::string session_obj(const std::string& sid, const SessionConfig& c) {
    std::ostringstream o;
    o << "{"
      << "\"id\":\"" << escape_json(sid) << "\","
      << "\"input_audio_format\":\"" << c.input_audio_format << "\","
      << "\"output_audio_format\":\"" << c.output_audio_format << "\","
      << "\"temperature\":" << c.temperature << ","
      << "\"top_k\":" << c.top_k
      << "}";
    return o.str();
}

// ---------------------------------------------------------------------------
// Serialisers
// ---------------------------------------------------------------------------
std::string make_session_created(const std::string& sid, const SessionConfig& c) {
    return "{\"type\":\"session.created\",\"session\":" + session_obj(sid, c) + "}";
}

std::string make_session_updated(const std::string& sid, const SessionConfig& c) {
    return "{\"type\":\"session.updated\",\"session\":" + session_obj(sid, c) + "}";
}

std::string make_session_ready(const std::string& sid) {
    return "{\"type\":\"session.ready\",\"session_id\":\"" + escape_json(sid) + "\"}";
}

std::string make_audio_delta(const std::string& resp_id,
                             const std::string& item_id,
                             const std::string& b64) {
    std::ostringstream o;
    o << "{\"type\":\"response.audio.delta\","
      << "\"response_id\":\"" << escape_json(resp_id) << "\","
      << "\"item_id\":\""     << escape_json(item_id) << "\","
      << "\"delta\":\""       << b64 << "\"}";
    return o.str();
}

std::string make_audio_done(const std::string& resp_id) {
    return "{\"type\":\"response.audio.done\",\"response_id\":\"" +
           escape_json(resp_id) + "\"}";
}

std::string make_text_delta(const std::string& resp_id, const std::string& delta) {
    std::ostringstream o;
    o << "{\"type\":\"response.text.delta\","
      << "\"response_id\":\"" << escape_json(resp_id) << "\","
      << "\"delta\":\""       << escape_json(delta)   << "\"}";
    return o.str();
}

std::string make_speech_started(int64_t ms) {
    return "{\"type\":\"input_audio_buffer.speech_started\","
           "\"audio_start_ms\":" + std::to_string(ms) + "}";
}

std::string make_speech_stopped(int64_t ms) {
    return "{\"type\":\"input_audio_buffer.speech_stopped\","
           "\"audio_end_ms\":" + std::to_string(ms) + "}";
}

std::string make_error(const std::string& type,
                       const std::string& msg,
                       const std::string& /*eid*/) {
    std::ostringstream o;
    o << "{\"type\":\"error\",\"error\":{\"type\":\""
      << escape_json(type) << "\",\"message\":\""
      << escape_json(msg)  << "\"}}";
    return o.str();
}

// ---------------------------------------------------------------------------
// Parser (simdjson on-demand API — zero-copy)
// ---------------------------------------------------------------------------
bool parse_client_event(const char* data, size_t len, ClientEvent& out) {
    static thread_local simdjson::ondemand::parser parser;

    simdjson::padded_string ps(data, len);
    auto doc = parser.iterate(ps);
    if (doc.error()) return false;

    std::string_view type_sv;
    if (doc["type"].get(type_sv) != simdjson::SUCCESS) return false;

    std::string type_str(type_sv);
    out.event_id = "";
    {
        std::string_view ev_sv;
        if (doc["event_id"].get(ev_sv) == simdjson::SUCCESS)
            out.event_id = std::string(ev_sv);
    }

    if (type_str == "session.update") {
        out.type = ClientEventType::SessionUpdate;
        auto sess = doc["session"];
        if (sess.error()) return false;

        std::string_view sv;
        if (sess["instructions"].get(sv) == simdjson::SUCCESS)
            out.session.instructions = std::string(sv);
        if (sess["voice_prompt_embedding"].get(sv) == simdjson::SUCCESS)
            out.session.voice_prompt_embedding = std::string(sv);
        if (sess["input_audio_format"].get(sv) == simdjson::SUCCESS)
            out.session.input_audio_format = std::string(sv);
        if (sess["output_audio_format"].get(sv) == simdjson::SUCCESS)
            out.session.output_audio_format = std::string(sv);

        double tmp;
        if (sess["temperature"].get(tmp) == simdjson::SUCCESS)
            out.session.temperature = static_cast<float>(tmp);
        int64_t itmp;
        if (sess["top_k"].get(itmp) == simdjson::SUCCESS)
            out.session.top_k = static_cast<int>(itmp);

        // Parse text_prompt_tokens array (used for voice name delivery)
        simdjson::ondemand::array tok_arr;
        if (sess["text_prompt_tokens"].get_array().get(tok_arr) == simdjson::SUCCESS) {
            for (auto val : tok_arr) {
                int64_t v;
                if (val.get_int64().get(v) == simdjson::SUCCESS)
                    out.session.text_prompt_tokens.push_back(static_cast<int32_t>(v));
            }
        }

    } else if (type_str == "input_audio_buffer.append") {
        out.type = ClientEventType::InputAudioBufferAppend;
        std::string_view sv;
        if (doc["audio"].get(sv) != simdjson::SUCCESS) return false;
        out.audio_b64 = std::string(sv);

    } else if (type_str == "input_audio_buffer.commit") {
        out.type = ClientEventType::InputAudioBufferCommit;
    } else if (type_str == "input_audio_buffer.clear") {
        out.type = ClientEventType::InputAudioBufferClear;
    } else if (type_str == "response.create") {
        out.type = ClientEventType::ResponseCreate;
    } else if (type_str == "response.cancel") {
        out.type = ClientEventType::ResponseCancel;
        std::string_view sv;
        if (doc["response_id"].get(sv) == simdjson::SUCCESS)
            out.response_id = std::string(sv);
    } else {
        out.type = ClientEventType::Unknown;
    }

    return true;
}

bool parse_client_event(const std::string& s, ClientEvent& out) {
    return parse_client_event(s.data(), s.size(), out);
}

} // namespace pg::proto
