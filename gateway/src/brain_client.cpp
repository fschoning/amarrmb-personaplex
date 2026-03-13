// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: MIT
//
// gateway/src/brain_client.cpp — HTTP client for the brain FastAPI server
//
// Uses libcurl for HTTP POST/GET.  The brain server runs at http://<host>:<port>
// and provides:
//   POST /generate  {"prompt": "...", "max_tokens": 64} → {"response": "..."}
//   GET  /health    → {"status": "ready"}

#include "brain_client.h"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <curl/curl.h>

namespace pg {

// ── curl write callback ─────────────────────────────────────────────────────
static size_t write_callback(void* contents, size_t size, size_t nmemb,
                             std::string* output) {
    size_t total = size * nmemb;
    output->append(static_cast<char*>(contents), total);
    return total;
}

// ── Constructor ─────────────────────────────────────────────────────────────
BrainClient::BrainClient(const std::string& url, const std::string& model_name)
    : model_name_(model_name)
{
    // Normalize URL: ensure http:// prefix
    if (url.find("://") == std::string::npos) {
        base_url_ = "http://" + url;
    } else {
        base_url_ = url;
    }
    // Remove trailing slash
    while (!base_url_.empty() && base_url_.back() == '/') {
        base_url_.pop_back();
    }
    fprintf(stderr, "[brain] HTTP client targeting %s\n", base_url_.c_str());
}

// ── is_ready ────────────────────────────────────────────────────────────────
bool BrainClient::is_ready() const {
    try {
        std::string resp = http_get(base_url_ + "/health", 5);
        return resp.find("\"ready\"") != std::string::npos;
    } catch (...) {
        return false;
    }
}

// ── query ───────────────────────────────────────────────────────────────────
std::string BrainClient::query(const std::string& prompt, int32_t max_tokens) {
    // Build JSON body.  Escape the prompt for JSON.
    std::string escaped;
    escaped.reserve(prompt.size() + 64);
    for (char c : prompt) {
        switch (c) {
            case '"':  escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\n': escaped += "\\n";  break;
            case '\r': escaped += "\\r";  break;
            case '\t': escaped += "\\t";  break;
            default:   escaped += c;      break;
        }
    }

    std::string body = "{\"prompt\":\"" + escaped + "\",\"max_tokens\":" +
                       std::to_string(max_tokens) + "}";

    try {
        std::string resp = http_post(base_url_ + "/generate", body, 120);

        // Parse "response" field from JSON — simple extraction
        // Looking for: "response":"<value>"
        const std::string key = "\"response\":\"";
        auto pos = resp.find(key);
        if (pos == std::string::npos) {
            // Try alternate format: "response": "..."
            const std::string key2 = "\"response\": \"";
            pos = resp.find(key2);
            if (pos == std::string::npos) {
                std::cerr << "[brain] response field not found in: "
                          << resp.substr(0, 200) << "\n";
                return "";
            }
            pos += key2.size();
        } else {
            pos += key.size();
        }

        // Find closing quote (handle escaped quotes)
        std::string result;
        for (size_t i = pos; i < resp.size(); ++i) {
            if (resp[i] == '\\' && i + 1 < resp.size()) {
                char next = resp[i + 1];
                if (next == '"')  { result += '"'; ++i; }
                else if (next == '\\') { result += '\\'; ++i; }
                else if (next == 'n')  { result += '\n'; ++i; }
                else if (next == 'r')  { result += '\r'; ++i; }
                else if (next == 't')  { result += '\t'; ++i; }
                else { result += resp[i]; }
            } else if (resp[i] == '"') {
                break;  // end of value
            } else {
                result += resp[i];
            }
        }
        return result;

    } catch (const std::exception& e) {
        std::cerr << "[brain] HTTP query failed: " << e.what() << "\n";
        return "";
    }
}

// ── HTTP POST ───────────────────────────────────────────────────────────────
std::string BrainClient::http_post(const std::string& url,
                                   const std::string& json_body,
                                   long timeout_s) {
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("curl_easy_init failed");

    std::string response;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_s);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);

    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        throw std::runtime_error(std::string("curl POST failed: ") +
                                 curl_easy_strerror(res));
    }
    return response;
}

// ── HTTP GET ────────────────────────────────────────────────────────────────
std::string BrainClient::http_get(const std::string& url, long timeout_s) {
    CURL* curl = curl_easy_init();
    if (!curl) throw std::runtime_error("curl_easy_init failed");

    std::string response;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout_s);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        throw std::runtime_error(std::string("curl GET failed: ") +
                                 curl_easy_strerror(res));
    }
    return response;
}

} // namespace pg
