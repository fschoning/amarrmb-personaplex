// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: MIT
//
// gateway/src/brain_client.cpp — Triton gRPC client for the "brain" LLM model

#include "brain_client.h"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace pg {

BrainClient::BrainClient(const std::string& triton_url,
                         const std::string& model_name)
    : model_name_(model_name)
{
    tc::Error err = tc::InferenceServerGrpcClient::Create(&client_, triton_url,
                                                          /*verbose=*/false);
    if (!err.IsOk()) {
        throw std::runtime_error("BrainClient: failed to connect to Triton at "
                                 + triton_url + ": " + err.Message());
    }
}

bool BrainClient::is_ready() const {
    bool ready = false;
    tc::Error err = client_->IsModelReady(&ready, model_name_);
    return err.IsOk() && ready;
}

std::string BrainClient::query(const std::string& prompt, int32_t max_tokens) {
    // ── PROMPT input ───────────────────────────────────────────────────────
    tc::InferInput* prompt_inp_raw = nullptr;
    std::vector<int64_t> prompt_shape = {1};
    tc::Error err = tc::InferInput::Create(&prompt_inp_raw, "PROMPT",
                                           prompt_shape, "BYTES");
    if (!err.IsOk()) {
        std::cerr << "[brain] InferInput PROMPT failed: " << err.Message() << "\n";
        return "";
    }
    std::shared_ptr<tc::InferInput> prompt_inp(prompt_inp_raw);

    // Triton BYTES: 4-byte little-endian length prefix + data
    uint32_t plen = static_cast<uint32_t>(prompt.size());
    std::vector<uint8_t> prompt_bytes(4 + prompt.size());
    memcpy(prompt_bytes.data(), &plen, 4);
    memcpy(prompt_bytes.data() + 4, prompt.data(), prompt.size());
    err = prompt_inp->AppendRaw(prompt_bytes.data(), prompt_bytes.size());
    if (!err.IsOk()) {
        std::cerr << "[brain] PROMPT AppendRaw failed: " << err.Message() << "\n";
        return "";
    }

    // ── MAX_TOKENS input ───────────────────────────────────────────────────
    tc::InferInput* max_tok_inp_raw = nullptr;
    std::vector<int64_t> max_tok_shape = {1};
    err = tc::InferInput::Create(&max_tok_inp_raw, "MAX_TOKENS",
                                 max_tok_shape, "INT32");
    if (!err.IsOk()) {
        std::cerr << "[brain] InferInput MAX_TOKENS failed: " << err.Message() << "\n";
        return "";
    }
    std::shared_ptr<tc::InferInput> max_tok_inp(max_tok_inp_raw);
    err = max_tok_inp->AppendRaw(
        reinterpret_cast<const uint8_t*>(&max_tokens), sizeof(int32_t));
    if (!err.IsOk()) {
        std::cerr << "[brain] MAX_TOKENS AppendRaw failed: " << err.Message() << "\n";
        return "";
    }

    // ── RESPONSE output ────────────────────────────────────────────────────
    tc::InferRequestedOutput* resp_out_raw = nullptr;
    err = tc::InferRequestedOutput::Create(&resp_out_raw, "RESPONSE");
    if (!err.IsOk()) {
        std::cerr << "[brain] InferRequestedOutput failed: " << err.Message() << "\n";
        return "";
    }
    std::shared_ptr<tc::InferRequestedOutput> resp_out(resp_out_raw);

    // ── Infer ──────────────────────────────────────────────────────────────
    std::vector<tc::InferInput*>            inputs  = {prompt_inp.get(), max_tok_inp.get()};
    std::vector<const tc::InferRequestedOutput*> outputs = {resp_out.get()};

    tc::InferOptions opts(model_name_);

    tc::InferResult* result_raw = nullptr;
    err = client_->Infer(&result_raw, opts, inputs, outputs);
    if (!err.IsOk()) {
        std::cerr << "[brain] Infer failed: " << err.Message() << "\n";
        return "";
    }
    std::shared_ptr<tc::InferResult> result(result_raw);

    // ── Decode BYTES output ────────────────────────────────────────────────
    std::vector<std::string> str_results;
    err = result->StringData("RESPONSE", &str_results);
    if (!err.IsOk() || str_results.empty()) {
        std::cerr << "[brain] StringData failed: " << err.Message() << "\n";
        return "";
    }
    return str_results[0];
}

} // namespace pg
