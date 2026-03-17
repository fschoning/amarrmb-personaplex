// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: MIT
//
// gateway/src/triton_client.cpp — Triton gRPC client wrapper

#include "triton_client.h"
#include "audio_utils.h"

#include <grpc_client.h>
#include <stdexcept>
#include <iostream>
#include <cstring>

namespace pg {

namespace {
    void check(const tc::Error& err, const char* ctx) {
        if (!err.IsOk())
            throw std::runtime_error(std::string(ctx) + ": " + err.Message());
    }
} // anon

TritonSession::TritonSession(const std::string& url,
                             const std::string& model,
                             int64_t            corrid,
                             int64_t            ver)
    : model_name_(model), corrid_(corrid), model_version_(ver)
{
    check(tc::InferenceServerGrpcClient::Create(&client_, url,
             /*verbose=*/false), "GrpcClient::Create");
}

TritonSession::~TritonSession() {
    if (stream_open_) {
        try { send_end(); } catch (...) {}
    }
}

// ---------------------------------------------------------------------------
// Internal: single synchronous inference request
// ---------------------------------------------------------------------------
bool TritonSession::infer_one(
    tc::InferInput*              pcm_in,
    bool                         is_start,
    bool                         is_end,
    const std::vector<uint8_t>*  voice_bytes,
    const std::vector<int32_t>*  text_tokens,
    FrameOutput&                 out)
{
    std::vector<tc::InferInput*>           inputs;
    std::vector<const tc::InferRequestedOutput*> outputs;

    // --- Control signals ---
    tc::InferInput* start_t = nullptr;
    tc::InferInput* end_t   = nullptr;
    tc::InferInput* corr_t  = nullptr;

    {
        float fstart = is_start ? 1.0f : 0.0f;
        float fend   = is_end   ? 1.0f : 0.0f;

        check(tc::InferInput::Create(&start_t, "START", {1}, "FP32"), "START");
        start_t->AppendRaw(reinterpret_cast<uint8_t*>(&fstart), sizeof(float));

        check(tc::InferInput::Create(&end_t,   "END",   {1}, "FP32"), "END");
        end_t->AppendRaw(reinterpret_cast<uint8_t*>(&fend),   sizeof(float));

        check(tc::InferInput::Create(&corr_t,  "CORRID",{1}, "INT64"), "CORRID");
        corr_t->AppendRaw(reinterpret_cast<uint8_t*>(&corrid_), sizeof(int64_t));
    }
    inputs.push_back(start_t);
    inputs.push_back(end_t);
    inputs.push_back(corr_t);

    // --- PCM frame (regular frames only) ---
    tc::InferInput* pcm_input = nullptr;
    if (pcm_in) {
        inputs.push_back(pcm_in);
    } else {
        // START/END: send a zero PCM frame as placeholder
        check(tc::InferInput::Create(&pcm_input, "INPUT_PCM",
              {1, 1, audio::FRAME_SAMPLES_24K}, "FP32"), "INPUT_PCM");
        std::vector<float> zeros(audio::FRAME_SAMPLES_24K, 0.f);
        pcm_input->AppendRaw(
            reinterpret_cast<const uint8_t*>(zeros.data()),
            zeros.size() * sizeof(float));
        inputs.push_back(pcm_input);
    }

    // --- Voice + text prompt (always sent; ensemble requires all inputs) ---
    tc::InferInput* vp_t  = nullptr;
    tc::InferInput* tp_t  = nullptr;
    {
        // Voice prompt bytes (TYPE_STRING, dims=[1] → exactly 1 string element)
        // Triton BYTES/STRING encoding: 4-byte little-endian length prefix + data
        check(tc::InferInput::Create(&vp_t, "VOICE_PROMPT_BYTES", {1}, "BYTES"), "VP");
        if (is_start && voice_bytes && !voice_bytes->empty()) {
            uint32_t sz = static_cast<uint32_t>(voice_bytes->size());
            vp_t->AppendRaw(reinterpret_cast<uint8_t*>(&sz), 4);
            vp_t->AppendRaw(voice_bytes->data(), voice_bytes->size());
        } else {
            // Empty string: length=0, no data bytes
            // Must use AppendFromString to get proper BYTES framing
            vp_t->AppendFromString({""});
        }
        inputs.push_back(vp_t);

        // Text prompt tokens
        size_t n_tok = (is_start && text_tokens) ? text_tokens->size() : 0;
        check(tc::InferInput::Create(&tp_t, "TEXT_PROMPT_TOKENS",
              {static_cast<int64_t>(std::max(n_tok, (size_t)1))}, "INT32"), "TP");
        if (n_tok > 0) {
            tp_t->AppendRaw(reinterpret_cast<const uint8_t*>(text_tokens->data()),
                            n_tok * sizeof(int32_t));
        } else {
            int32_t zero = 0;
            tp_t->AppendRaw(reinterpret_cast<uint8_t*>(&zero), sizeof(int32_t));
        }
        inputs.push_back(tp_t);
    }

    // --- Requested outputs ---
    tc::InferRequestedOutput* out_pcm = nullptr;
    tc::InferRequestedOutput* out_tok = nullptr;
    tc::InferRequestedOutput* out_txt = nullptr;
    tc::InferRequestedOutput* out_rdy = nullptr;
    check(tc::InferRequestedOutput::Create(&out_pcm, "OUTPUT_PCM_48K"), "req_pcm");
    check(tc::InferRequestedOutput::Create(&out_tok, "TEXT_TOKEN"),      "req_tok");
    check(tc::InferRequestedOutput::Create(&out_txt, "TEXT_DECODED"),    "req_txt");
    check(tc::InferRequestedOutput::Create(&out_rdy, "SESSION_READY"),   "req_rdy");
    outputs.push_back(out_pcm);
    outputs.push_back(out_tok);
    outputs.push_back(out_txt);
    outputs.push_back(out_rdy);

    // --- Infer ---
    tc::InferOptions opts(model_name_);
    opts.model_version_       = (model_version_ == -1) ? "" : std::to_string(model_version_);
    opts.sequence_id_         = static_cast<uint64_t>(corrid_);
    opts.sequence_start_      = is_start;
    opts.sequence_end_        = is_end;

    tc::InferResult* result = nullptr;
    auto err = client_->Infer(&result, opts, inputs, outputs);

    // Cleanup input objects
    for (auto* inp : inputs) delete inp;
    for (auto* o   : outputs) delete o;

    if (!err.IsOk()) {
        std::cerr << "[triton] Infer error: " << err.Message() << "\n";
        return false;
    }

    std::unique_ptr<tc::InferResult> res(result);

    // --- Decode outputs ---
    if (!is_end) {
        // PCM_48K
        const float* pcm_ptr = nullptr;
        size_t       pcm_bytes = 0;
        res->RawData("OUTPUT_PCM_48K",
            reinterpret_cast<const uint8_t**>(&pcm_ptr), &pcm_bytes);
        size_t n_pcm = pcm_bytes / sizeof(float);
        out.pcm_48k.assign(pcm_ptr, pcm_ptr + n_pcm);

        // TEXT_TOKEN
        const int32_t* tok_ptr = nullptr;
        size_t tok_bytes = 0;
        res->RawData("TEXT_TOKEN",
            reinterpret_cast<const uint8_t**>(&tok_ptr), &tok_bytes);
        out.text_token = tok_bytes >= 4 ? tok_ptr[0] : 0;

        // TEXT_DECODED (Triton BYTES: 4-byte length prefix + string data)
        const uint8_t* td_raw = nullptr;
        size_t td_bytes = 0;
        auto td_err = res->RawData("TEXT_DECODED", &td_raw, &td_bytes);
        if (td_err.IsOk() && td_bytes > 4) {
            uint32_t str_len = 0;
            std::memcpy(&str_len, td_raw, 4);
            if (str_len > 0 && 4 + str_len <= td_bytes) {
                out.text_decoded.assign(
                    reinterpret_cast<const char*>(td_raw + 4), str_len);
            }
        }

        // SESSION_READY
        const bool* rdy_ptr = nullptr;
        size_t rdy_bytes = 0;
        res->RawData("SESSION_READY",
            reinterpret_cast<const uint8_t**>(&rdy_ptr), &rdy_bytes);
        out.session_ready = rdy_bytes > 0 && rdy_ptr[0];
    }

    return true;
}

bool TritonSession::send_start(const std::vector<uint8_t>& voice_bytes,
                                const std::vector<int32_t>& text_tokens) {
    FrameOutput out;
    bool ok = infer_one(nullptr, /*start=*/true, /*end=*/false,
                        &voice_bytes, &text_tokens, out);
    if (ok) stream_open_ = true;
    return ok;
}

bool TritonSession::send_frame(const float* pcm_24k, size_t n, FrameOutput& out) {
    tc::InferInput* pcm_in = nullptr;
    check(tc::InferInput::Create(&pcm_in, "INPUT_PCM",
          {1, 1, static_cast<int64_t>(n)}, "FP32"), "INPUT_PCM frame");
    pcm_in->AppendRaw(reinterpret_cast<const uint8_t*>(pcm_24k), n * sizeof(float));

    return infer_one(pcm_in, /*start=*/false, /*end=*/false, nullptr, nullptr, out);
}

bool TritonSession::send_end() {
    FrameOutput out;
    bool ok = infer_one(nullptr, /*start=*/false, /*end=*/true, nullptr, nullptr, out);
    stream_open_ = false;
    return ok;
}

} // namespace pg
