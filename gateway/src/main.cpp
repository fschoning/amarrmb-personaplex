// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: MIT
//
// gateway/src/main.cpp — PersonaPlex v3 Gateway (Client-Driven)
//
// Architecture:
//   uWS event loop thread:
//     onOpen   → create Session, start session worker thread
//     onMessage→ parse JSON event; audio → ring buffer;
//                node.* / session.configure → orchestrator command_handler
//                session.update → set config, signal worker
//     onClose  → signal worker, SessionManager.remove()
//
//   Session worker thread (one per active session):
//     1. Wait for config (session.update from client)
//     2. Create Orchestrator (Active + Standby + Filler PP sessions)
//     3. Orchestrator.run() drives the audio loop until disconnect

#include "config.h"
#include "orchestrator.h"
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
// Session worker: creates and drives the Orchestrator for one session
// ---------------------------------------------------------------------------
static void session_worker(std::shared_ptr<Session> sess,
                            const Config& cfg,
                            uWS::Loop* loop)
{
    // Helper: safely post a JSON string to the client via the uWS event loop
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

    fprintf(stderr, "[worker] session %s: starting Orchestrator (voice_bytes=%zu, persona='%.40s')\n",
            sess->session_id.c_str(),
            sess->config.voice_prompt_embedding.size(),
            sess->config.persona_prompt.c_str());

    // --- 2. Run the Orchestrator (blocks until session ends) ---
    try {
        Orchestrator orch(cfg, sess, send);
        orch.run();
    } catch (const std::exception& e) {
        fprintf(stderr, "[worker] session %s: Orchestrator exception: %s\n",
                sess->session_id.c_str(), e.what());
        send(make_error("server_error", std::string("Orchestrator: ") + e.what(), ""));
    }
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

    std::cout << "PersonaPlex Gateway v3\n"
              << "  WebSocket port : " << cfg.ws_port       << "\n"
              << "  Triton URL     : " << cfg.triton_url    << "\n"
              << "  Max sessions   : " << cfg.max_sessions  << "\n"
              << "  TLS            : " << (cfg.ssl_cert.empty() ? "off" : "on") << "\n"
              << "  Nodes          : Active + Standby + Filler (3 PP instances)\n"
              << "  Brain          : client-side (workstation)\n";

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
                case ClientEventType::NodeCommand: {
                    // Route node.* and session.configure to orchestrator
                    std::lock_guard<std::mutex> lk(sess->command_handler_mtx);
                    if (sess->command_handler) {
                        sess->command_handler(std::string(msg.data(), msg.size()));
                    }
                    break;
                }
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
