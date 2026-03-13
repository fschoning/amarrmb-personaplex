// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: MIT
//
// gateway/src/main.cpp — PersonaPlex v2 Gateway
//
// Architecture:
//   uWS event loop thread:
//     onOpen   → create Session, start session worker thread
//     onMessage→ parse JSON event; audio → push to ring buffer;
//                session.update → set config, signal worker
//     onClose  → signal worker, SessionManager.remove()
//
//   Session worker thread (one per active session):
//     1. Wait for config (session.update from client)
//     2. decode voice_prompt_embedding (base64 → bytes)
//     3. send_start() to Triton  <-- blocks ~2-5s (system prompts)
//     4. send session.ready to client via loop->defer()
//     5. Loop: pop 1920 samples from ring buffer → send_frame() → encode_audio_delta()
//     6. send_end(), exit.

#include "config.h"
#include "session.h"
#include "protocol.h"
#include "triton_client.h"
#include "audio_utils.h"

// uWebSockets (header-only, v20)
#include <App.h>

#include <atomic>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace pg;
using namespace pg::audio;
using namespace pg::proto;

// Cleanly shut down on Ctrl-C
static std::atomic<bool> g_stop{false};
static uWS::Loop*        g_loop = nullptr;

static void sighandler(int) {
    g_stop.store(true);
    if (g_loop) g_loop->defer([] { /* wake loop so it checks g_stop */ });
}

// ---------------------------------------------------------------------------
// Session worker: runs Triton inference for one session
// ---------------------------------------------------------------------------
static void session_worker(std::shared_ptr<Session> sess,
                            const Config& cfg,
                            uWS::Loop* loop)
{
    // Helper: safely post a JSON string to the client
    // Must go via loop->defer() because uWS is single-threaded.
    auto send = [&](std::string msg) {
        std::string m = std::move(msg);
        loop->defer([sess, m = std::move(m)]() mutable {
            sess->send_to_client(std::move(m));
        });
    };

    // --- 1. Wait for session.update (config_set) ---
    while (!sess->config_set && !sess->should_close.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

    if (sess->should_close.load()) return;

    // --- 2. Decode voice prompt bytes ---
    std::vector<uint8_t> voice_bytes;
    if (!sess->config.voice_prompt_embedding.empty()) {
        voice_bytes = base64_decode(sess->config.voice_prompt_embedding);
    }

    // Text prompt tokens (may contain voice name encoded as int32 sentinel)
    std::vector<int32_t> text_tokens = sess->config.text_prompt_tokens;

    fprintf(stderr, "[worker] session %s: voice_bytes=%zu text_tokens=%zu\n",
            sess->session_id.c_str(), voice_bytes.size(), text_tokens.size());

    // --- 3. Triton START (blocks during system prompt conditioning) ---
    std::string resp_id  = "resp_" + sess->session_id;
    std::string item_id  = "item_" + sess->session_id;

    TritonSession ts(cfg.triton_url, cfg.pipeline_model, sess->corrid, cfg.model_version);

    fprintf(stderr, "[worker] session %s: sending START to Triton...\n",
            sess->session_id.c_str());

    try {
        bool ok = ts.send_start(voice_bytes, text_tokens);
        fprintf(stderr, "[worker] session %s: Triton START returned ok=%d\n",
                sess->session_id.c_str(), ok);
        if (!ok || sess->should_close.load()) {
            send(make_error("server_error", "Triton start failed", ""));
            return;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "[worker] session %s: Triton exception: %s\n",
                sess->session_id.c_str(), e.what());
        send(make_error("server_error", std::string("Triton: ") + e.what(), ""));
        return;
    }

    // --- 4. Signal client that we are ready ---
    sess->triton_ready.store(true);
    send(make_session_ready(sess->session_id));

    // --- 5. Inference loop (80ms frames) ---
    std::vector<float> frame_buf(FRAME_SAMPLES_24K);

    while (!sess->should_close.load()) {
        size_t got = sess->audio_in.pop(frame_buf.data(), FRAME_SAMPLES_24K, /*timeout_ms=*/200);
        if (got == 0) continue;  // timeout — try again (allows checking should_close)

        FrameOutput out;
        bool ok = false;
        try {
            ok = ts.send_frame(frame_buf.data(), FRAME_SAMPLES_24K, out);
        } catch (const std::exception& e) {
            std::cerr << "[worker:" << sess->session_id << "] Triton error: " << e.what() << "\n";
            break;
        }

        if (!ok) break;

        // Audio delta — 48kHz PCM float32 → base64 PCM16
        if (!out.pcm_48k.empty()) {
            std::string b64 = encode_audio_delta(out.pcm_48k.data(), out.pcm_48k.size());
            send(make_audio_delta(resp_id, item_id, b64));
        }

        // Text token — log for Phase 0 transcript analysis.
        // Log every token (including 0) for every 25th frame, plus non-zero always.
        static int frame_count = 0;
        ++frame_count;
        if (out.text_token > 0 && out.text_token < 32000) {
            fprintf(stderr, "[tok] %.8s %d\n",
                    sess->session_id.c_str(), out.text_token);
        } else if (frame_count % 25 == 0) {
            fprintf(stderr, "[tok-dbg] %.8s frame=%d token=%d\n",
                    sess->session_id.c_str(), frame_count, out.text_token);
        }
    }

    // --- 6. Clean END ---
    try { ts.send_end(); } catch (...) {}
    send(make_audio_done(resp_id));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    Config cfg = Config::from_env();

    // Allow CLI override for port
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--port")   cfg.ws_port  = (uint16_t)std::atoi(argv[i+1]);
        if (std::string(argv[i]) == "--triton")  cfg.triton_url = argv[i+1];
        if (std::string(argv[i]) == "--sessions") cfg.max_sessions = std::atoi(argv[i+1]);
    }

    std::cout << "PersonaPlex Gateway v2\n"
              << "  WebSocket port : " << cfg.ws_port       << "\n"
              << "  Triton URL     : " << cfg.triton_url    << "\n"
              << "  Max sessions   : " << cfg.max_sessions  << "\n"
              << "  TLS            : " << (cfg.ssl_cert.empty() ? "off" : "on") << "\n";

    std::signal(SIGINT,  sighandler);
    std::signal(SIGTERM, sighandler);

    SessionManager sm(cfg);

    // Per-userData stored on the WebSocket socket
    struct WsData { void* key = nullptr; };

    auto make_app = [&]() -> uWS::App {
        return uWS::App()
        .ws<WsData>("/v1/realtime", {
            // --- Settings ---
            .compression      = uWS::DISABLED,       // raw PCM is incompressible
            .maxPayloadLength = 16 * 1024 * 1024,    // 16MB max message
            .idleTimeout      = static_cast<unsigned short>(cfg.session_timeout_s),

            // --- onOpen ---
            .open = [&](uWS::WebSocket<false, true, WsData>* ws) {
                void* key = ws;
                uWS::Loop* loop = uWS::Loop::get();

                auto send_fn = [ws, loop](std::string msg) {
                    // This lambda is called from worker threads via loop->defer().
                    // The defer'd closure runs on the uWS thread, where ws is valid
                    // as long as onClose hasn't fired. uWS guarantees the deferred
                    // call won't race with onClose — both run on the same event loop.
                    ws->send(msg, uWS::OpCode::TEXT);
                };

                auto sess = sm.create(key, std::move(send_fn));
                if (!sess) {
                    ws->send(make_error("capacity_exceeded",
                        "Maximum concurrent sessions reached. Try again later."),
                        uWS::OpCode::TEXT);
                    ws->close();
                    return;
                }

                ws->getUserData()->key = key;
                g_loop = loop;

                // Send session.created immediately
                ws->send(make_session_created(sess->session_id, sess->config),
                         uWS::OpCode::TEXT);

                // Start worker thread
                sess->worker = std::thread(session_worker, sess, std::cref(cfg), loop);
            },

            // --- onMessage ---
            .message = [&](uWS::WebSocket<false, true, WsData>* ws,
                           std::string_view msg,
                           uWS::OpCode /*opcode*/)
            {
                void* key = ws->getUserData()->key;
                auto sess = sm.get(key);
                if (!sess) return;

                ClientEvent ev;
                if (!parse_client_event(msg.data(), msg.size(), ev)) {
                    ws->send(make_error("invalid_request", "JSON parse error"),
                             uWS::OpCode::TEXT);
                    return;
                }

                switch (ev.type) {
                case ClientEventType::SessionUpdate: {
                    sess->config = ev.session;
                    sess->config_set = true;
                    ws->send(make_session_updated(sess->session_id, sess->config),
                             uWS::OpCode::TEXT);
                    break;
                }
                case ClientEventType::InputAudioBufferAppend: {
                    if (!sess->triton_ready.load()) break; // discard before ready
                    // Decode base64 PCM16 → float32 into ring buffer
                    static thread_local std::vector<float> tmp;
                    tmp.clear();
                    size_t n = decode_audio_event(ev.audio_b64, tmp);
                    if (n > 0) sess->audio_in.push(tmp.data(), n);
                    break;
                }
                case ClientEventType::InputAudioBufferClear:
                    sess->audio_in.reset();
                    break;

                case ClientEventType::ResponseCancel:
                    sess->should_close.store(true);
                    break;

                default:
                    break;
                }
            },

            // --- onClose ---
            .close = [&](uWS::WebSocket<false, true, WsData>* ws, int /*code*/, std::string_view) {
                sm.remove(ws->getUserData()->key);
            }
        })
        .listen(cfg.ws_port, [&](auto* tok) {
            if (tok) {
                std::cout << "Listening on ws://0.0.0.0:" << cfg.ws_port
                          << "/v1/realtime\n";
            } else {
                std::cerr << "Failed to bind port " << cfg.ws_port << "\n";
                std::exit(1);
            }
        });
    };

    // Run the event loop
    make_app().run();
    return 0;
}
