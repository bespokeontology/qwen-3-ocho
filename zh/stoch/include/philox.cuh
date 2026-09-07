// philox.cuh — Philox4x32-10 counter-based RNG, implementing the frozen KD GEN1 §7
// contract (KD_STOCHASTIC_GEN1_SPEC.md, 2026-09-04):
//   key     = (slot, round)         -> (seed_lo = slot, seed_hi = round)
//   counter = (position, branch, purpose, index)
//             position = step, branch = branch id, purpose = DRAFT(1)/ACCEPT(2)/
//             RESIDUAL(3)/BONUS(4), index = lane
//   uniform = (word0 >> 8) * 2^-24   (24-bit float in [0,1))
// Round structure is the Random123 reference verbatim (sampled_mode/mcsd_reference.py,
// philox4x32_10): each of the 10 rounds multiplies c0*M0 and c2*M1 (64-bit), and the
// Weyl key bump is applied after every round except the last.
// KAT (GEN1 §7 / Random123): ctr = key = 0 -> 6627e8d5 e169c58d bc57ac4c 9b00dbd8.
#pragma once
#include <cstdint>

struct ph4x32_ctr { uint32_t v[4]; };
struct ph4x32_key { uint32_t v[2]; };

__host__ __device__ inline ph4x32_ctr ph4x32_round(ph4x32_ctr c, ph4x32_key k) {
    const uint64_t p0 = (uint64_t)c.v[0] * 0xD2511F53ull;   // M0
    const uint64_t p1 = (uint64_t)c.v[2] * 0xCD9E8D57ull;   // M1
    ph4x32_ctr r;
    r.v[0] = (uint32_t)(p1 >> 32) ^ c.v[1] ^ k.v[0];
    r.v[1] = (uint32_t)p1;
    r.v[2] = (uint32_t)(p0 >> 32) ^ c.v[3] ^ k.v[1];
    r.v[3] = (uint32_t)p0;
    return r;
}

__host__ __device__ inline ph4x32_ctr philox4x32_10(ph4x32_ctr ctr, ph4x32_key key) {
    for (int r = 0; r < 10; r++) {
        ctr = ph4x32_round(ctr, key);
        if (r != 9) { key.v[0] += 0x9E3779B9u; key.v[1] += 0xBB67AE85u; }
    }
    return ctr;
}

// Engine mapping: one (seed, branch, step, lane) cell -> one deterministic 4-word block.
__host__ __device__ inline ph4x32_ctr stoch_rand4(uint32_t seed_lo, uint32_t seed_hi,
                                                  uint32_t step, uint32_t branch,
                                                  uint32_t purpose, uint32_t lane) {
    ph4x32_key k; k.v[0] = seed_lo; k.v[1] = seed_hi;
    ph4x32_ctr c; c.v[0] = step; c.v[1] = branch; c.v[2] = purpose; c.v[3] = lane;
    return philox4x32_10(c, k);
}

// GEN1 §7 uniform extractor: 24-bit float in [0,1).
__host__ __device__ inline float stoch_uniform_u32(uint32_t word) {
    return (float)(word >> 8) * (1.0f / 16777216.0f);
}
