// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: MIT
//
// gateway/src/session.cpp

#include "session.h"

#include <cassert>
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>

namespace pg {

// ---------------------------------------------------------------------------
// AudioRingBuffer
// ---------------------------------------------------------------------------
AudioRingBuffer::AudioRingBuffer(size_t capacity)
    : buf_(capacity), capacity_(capacity) {}

size_t AudioRingBuffer::push(const float* samples, size_t n) {
    std::unique_lock<std::mutex> lock(mtx_);
    size_t space = capacity_ - count_;
    size_t to_write = std::min(n, space);
    for (size_t i = 0; i < to_write; ++i) {
        buf_[tail_] = samples[i];
        tail_ = (tail_ + 1) % capacity_;
    }
    count_ += to_write;
    cv_.notify_one();
    return to_write;
}

size_t AudioRingBuffer::pop(float* out, size_t n, int timeout_ms) {
    using namespace std::chrono_literals;
    std::unique_lock<std::mutex> lock(mtx_);
    bool ok = cv_.wait_for(lock,
        std::chrono::milliseconds(timeout_ms),
        [this, n] { return count_ >= n; });
    if (!ok) return 0;
    for (size_t i = 0; i < n; ++i) {
        out[i] = buf_[head_];
        head_ = (head_ + 1) % capacity_;
    }
    count_ -= n;
    return n;
}

void AudioRingBuffer::reset() {
    std::lock_guard<std::mutex> lock(mtx_);
    head_ = tail_ = count_ = 0;
}

size_t AudioRingBuffer::available() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return count_;
}

// ---------------------------------------------------------------------------
// UUID generation (simple random, not crypto-quality)
// ---------------------------------------------------------------------------
static std::string make_uuid() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<uint32_t> d;
    auto r = [&] { return d(rng); };
    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    ss << std::setw(8) << r() << '-'
       << std::setw(4) << (r() & 0xffff) << '-'
       << std::setw(4) << ((r() & 0x0fff) | 0x4000) << '-'
       << std::setw(4) << ((r() & 0x3fff) | 0x8000) << '-'
       << std::setw(8) << r()
       << std::setw(4) << (r() & 0xffff);
    return ss.str();
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------
Session::Session(int64_t id, std::string sid, int buf_ms)
    : corrid(id)
    , session_id(std::move(sid))
    , audio_in(
        static_cast<size_t>(audio::INPUT_SAMPLE_RATE) *
        static_cast<size_t>(buf_ms) / 1000)
{}

Session::~Session() {
    should_close.store(true);
    if (worker.joinable()) worker.join();
}

// ---------------------------------------------------------------------------
// SessionManager
// ---------------------------------------------------------------------------
SessionManager::SessionManager(const Config& cfg) : cfg_(cfg) {}

std::shared_ptr<Session> SessionManager::create(
    void* ws_key,
    std::function<void(std::string)> send_fn)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if ((int)sessions_.size() >= cfg_.max_sessions) return nullptr;

    int64_t id  = next_corrid_.fetch_add(1, std::memory_order_relaxed);
    auto sid    = make_uuid();
    auto sess   = std::make_shared<Session>(id, sid, cfg_.audio_buffer_ms);
    sess->send_to_client = std::move(send_fn);
    sessions_[ws_key] = sess;
    return sess;
}

std::shared_ptr<Session> SessionManager::get(void* ws_key) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = sessions_.find(ws_key);
    return it != sessions_.end() ? it->second : nullptr;
}

void SessionManager::remove(void* ws_key) {
    std::shared_ptr<Session> sess;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = sessions_.find(ws_key);
        if (it == sessions_.end()) return;
        sess = std::move(it->second);
        sessions_.erase(it);
    }
    // Signal and join outside the lock to avoid deadlocks
    if (sess) {
        sess->should_close.store(true);
        // Worker reads from audio_in; prod an empty push to unblock cv_.wait_for
        float dummy = 0;
        sess->audio_in.push(&dummy, 1);
    }
}

int SessionManager::active_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return (int)sessions_.size();
}

} // namespace pg
