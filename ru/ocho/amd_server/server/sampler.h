// sampler.h - host-side sampling from the fp32 logits vector.
//
// This is the sanctioned sampling path: the engine produces one QF4_NVOCAB
// fp32 logits vector per generated token (one D2H), and sampling happens on
// the host. Greedy when temperature == 0; otherwise temperature scaling +
// optional top_k + optional top_p (nucleus) + optional repetition penalty
// over the tokens seen so far in the request. RNG is xorshift128+ seeded
// per request (splitmix64 expand), so a given seed reproduces a request
// exactly. No <random>, no external deps.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace qf {

struct SamplerParams {
    float   temperature = 1.0f;        // 0 => greedy argmax
    int     top_k = 0;                 // 0 => off
    float   top_p = 1.0f;              // 1.0 => off
    float   repetition_penalty = 1.0f; // 1.0 => off (HF convention)
    uint64_t seed = 0;                 // used only when has_seed
    bool    has_seed = false;
};

class Sampler {
public:
    explicit Sampler(const SamplerParams &p);

    // logits: n fp32. seen: prompt + generated token ids so far in this
    // request (repetition penalty source). Returns the sampled token id.
    int sample(const float *logits, size_t n, const std::vector<int32_t> &seen);

private:
    SamplerParams p_;
    uint64_t s_[2];             // xorshift128+ state
    std::vector<float> work_;   // penalized/scaled logits scratch
    std::vector<int32_t> cand_; // candidate indices

    float uniform();            // (0,1]
};

} // namespace qf
