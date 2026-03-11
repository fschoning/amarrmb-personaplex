// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: MIT
//
// gateway/src/audio_utils.h — PCM conversion and Base64 encode/decode

#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pg::audio {

// ---------------------------------------------------------------------------
// Base64 (RFC 4648, no line breaks)
// Using AVX2/NEON-friendly lookup-table implementation in .cpp
// ---------------------------------------------------------------------------
std::string base64_encode(const uint8_t* data, size_t len);
std::string base64_encode(const std::vector<uint8_t>& data);

// Returns empty vector on bad input.
std::vector<uint8_t> base64_decode(const std::string& encoded);
std::vector<uint8_t> base64_decode(const char* data, size_t len);

// ---------------------------------------------------------------------------
// PCM helpers
// ---------------------------------------------------------------------------

/// Convert int16_t LE samples → float32 normalised to [-1, 1]
void pcm16_to_float32(const int16_t* src, float* dst, size_t n_samples);
void pcm16_to_float32(const uint8_t* src_bytes, float* dst, size_t n_samples);

/// Convert float32 [-1, 1] → int16_t LE (clamped)
void float32_to_pcm16(const float* src, int16_t* dst, size_t n_samples);
std::vector<uint8_t> float32_to_pcm16_bytes(const float* src, size_t n_samples);

/// Convert base64-encoded PCM16 string → float32 vector
/// Returns number of samples decoded, or 0 on error.
size_t decode_audio_event(const std::string& b64, std::vector<float>& out_samples);

/// Convert float32 48kHz PCM → base64-encoded PCM16 string
std::string encode_audio_delta(const float* samples_48k, size_t n_samples);

constexpr int INPUT_SAMPLE_RATE  = 24000;
constexpr int OUTPUT_SAMPLE_RATE = 48000;
constexpr int FRAME_SAMPLES_24K  = 1920;  // 80ms at 24kHz
constexpr int FRAME_SAMPLES_48K  = 3840;  // 80ms at 48kHz

} // namespace pg::audio
