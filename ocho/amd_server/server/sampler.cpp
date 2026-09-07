// sampler.cpp - host-side sampling. See sampler.h.
#include "sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <unistd.h>

namespace qf {

namespace {

uint64_t splitmix64(uint64_t &x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

} // namespace

Sampler::Sampler(const SamplerParams &p) : p_(p) {
    uint64_t seed = p_.has_seed
        ? p_.seed
        : (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count()
          ^ ((uint64_t)getpid() << 32);
    s_[0] = splitmix64(seed);
    s_[1] = splitmix64(seed);
    if (s_[0] == 0 && s_[1] == 0) s_[1] = 1;
}

float Sampler::uniform() {
    // xorshift128+
    uint64_t x = s_[0];
    uint64_t y = s_[1];
    s_[0] = y;
    x ^= x << 23;
    s_[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
    uint64_t r = s_[1] + y;
    return (float)((r >> 11) * (1.0 / 9007199254740992.0)) + 1e-7f;  // (0,1]
}

int Sampler::sample(const float *logits, size_t n, const std::vector<int32_t> &seen) {
    work_.assign(logits, logits + n);

    // Repetition penalty (HF convention): divide positive logits, multiply
    // negative ones, for every token id seen so far in this request.
    if (p_.repetition_penalty != 1.0f) {
        for (int32_t id : seen) {
            if (id < 0 || (size_t)id >= n) continue;
            float &l = work_[(size_t)id];
            l = l > 0.0f ? l / p_.repetition_penalty : l * p_.repetition_penalty;
        }
    }

    // Greedy.
    if (p_.temperature <= 0.0f) {
        return (int)(std::max_element(work_.begin(), work_.end()) - work_.begin());
    }

    float inv_t = 1.0f / p_.temperature;
    for (size_t i = 0; i < n; i++) work_[i] *= inv_t;

    // Candidate set: top_k via nth_element, else everything.
    cand_.resize(n);
    for (size_t i = 0; i < n; i++) cand_[i] = (int32_t)i;
    auto better = [&](int32_t a, int32_t b) { return work_[(size_t)a] > work_[(size_t)b]; };
    if (p_.top_k > 0 && (size_t)p_.top_k < n) {
        std::nth_element(cand_.begin(), cand_.begin() + p_.top_k, cand_.end(), better);
        cand_.resize((size_t)p_.top_k);
    }

    // Softmax over candidates.
    float m = work_[(size_t)cand_[0]];
    for (int32_t c : cand_) m = std::max(m, work_[(size_t)c]);
    double sum = 0.0;
    for (int32_t c : cand_) sum += std::exp((double)work_[(size_t)c] - (double)m);

    // Nucleus: shrink the candidate set to the smallest prefix (sorted by
    // descending probability) whose cumulative mass reaches top_p.
    if (p_.top_p < 1.0f && cand_.size() > 1) {
        std::sort(cand_.begin(), cand_.end(), better);
        double acc = 0.0, cutoff = (double)p_.top_p * sum;
        size_t keep = 1;
        for (size_t i = 0; i < cand_.size(); i++) {
            acc += std::exp((double)work_[(size_t)cand_[i]] - (double)m);
            keep = i + 1;
            if (acc >= cutoff) break;
        }
        cand_.resize(keep);
        // Recompute mass over the kept prefix.
        sum = 0.0;
        for (int32_t c : cand_) sum += std::exp((double)work_[(size_t)c] - (double)m);
    }

    double r = (double)uniform() * sum;
    double acc = 0.0;
    for (int32_t c : cand_) {
        acc += std::exp((double)work_[(size_t)c] - (double)m);
        if (r <= acc) return c;
    }
    return cand_.back();
}

} // namespace qf
