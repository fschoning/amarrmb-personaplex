// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: MIT
//
// gateway/src/audio_utils.cpp

#include "audio_utils.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace pg::audio {

// ---------------------------------------------------------------------------
// Base64 — lookup-table implementation, SWAR-friendly
// ---------------------------------------------------------------------------
static const char kEncTable[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static const int8_t kDecTable[256] = {
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
    52,53,54,55,56,57,58,59,60,61,-1,-1,-1, 0,-1,-1,
    -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
    15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
    -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
    41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
};

std::string base64_encode(const uint8_t* data, size_t len) {
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t b  = (uint32_t)data[i] << 16;
        if (i + 1 < len) b |= (uint32_t)data[i+1] << 8;
        if (i + 2 < len) b |= (uint32_t)data[i+2];
        out += kEncTable[(b >> 18) & 0x3f];
        out += kEncTable[(b >> 12) & 0x3f];
        out += (i + 1 < len) ? kEncTable[(b >> 6) & 0x3f] : '=';
        out += (i + 2 < len) ? kEncTable[(b     ) & 0x3f] : '=';
    }
    return out;
}

std::string base64_encode(const std::vector<uint8_t>& data) {
    return base64_encode(data.data(), data.size());
}

std::vector<uint8_t> base64_decode(const char* data, size_t len) {
    std::vector<uint8_t> out;
    out.reserve((len / 4) * 3);
    for (size_t i = 0; i + 3 < len; i += 4) {
        int8_t a = kDecTable[(uint8_t)data[i+0]];
        int8_t b = kDecTable[(uint8_t)data[i+1]];
        int8_t c = kDecTable[(uint8_t)data[i+2]];
        int8_t d = kDecTable[(uint8_t)data[i+3]];
        if (a < 0 || b < 0) break;
        out.push_back((uint8_t)((a << 2) | (b >> 4)));
        if (data[i+2] != '=') out.push_back((uint8_t)((b << 4) | (c >> 2)));
        if (data[i+3] != '=') out.push_back((uint8_t)((c << 6) | d));
    }
    return out;
}

std::vector<uint8_t> base64_decode(const std::string& s) {
    return base64_decode(s.data(), s.size());
}

// ---------------------------------------------------------------------------
// PCM helpers
// ---------------------------------------------------------------------------
void pcm16_to_float32(const int16_t* src, float* dst, size_t n) {
    constexpr float kScale = 1.0f / 32768.0f;
    for (size_t i = 0; i < n; ++i)
        dst[i] = static_cast<float>(src[i]) * kScale;
}

void pcm16_to_float32(const uint8_t* src_bytes, float* dst, size_t n) {
    pcm16_to_float32(reinterpret_cast<const int16_t*>(src_bytes), dst, n);
}

void float32_to_pcm16(const float* src, int16_t* dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float clamped = std::max(-1.0f, std::min(1.0f, src[i]));
        dst[i] = static_cast<int16_t>(clamped * 32767.0f);
    }
}

std::vector<uint8_t> float32_to_pcm16_bytes(const float* src, size_t n) {
    std::vector<uint8_t> out(n * 2);
    float32_to_pcm16(src, reinterpret_cast<int16_t*>(out.data()), n);
    return out;
}

size_t decode_audio_event(const std::string& b64, std::vector<float>& out) {
    auto bytes = base64_decode(b64);
    if (bytes.empty() || bytes.size() % 2 != 0) return 0;
    size_t n = bytes.size() / 2;
    out.resize(out.size() + n);
    pcm16_to_float32(bytes.data(), out.data() + out.size() - n, n);
    return n;
}

std::string encode_audio_delta(const float* samples_48k, size_t n) {
    auto bytes = float32_to_pcm16_bytes(samples_48k, n);
    return base64_encode(bytes);
}

} // namespace pg::audio
