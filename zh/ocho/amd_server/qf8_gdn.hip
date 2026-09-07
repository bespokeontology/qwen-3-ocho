// qf8_gdn.hip - fused GDN (gated delta net / linear attention) decode mixer
// for gfx906. T=1 only.
//
// One fused kernel replaces the old 4-launch sequence
// (k_conv_step + k_gdn_decode + k_rmsnorm_gated + k_cvt_bf16_f32):
//
//   grid 16 blocks x 192 threads (3 waves of 64). Block kh owns k-head kh and
//   its 3 v-heads, and -- crucially -- exactly the conv channels of those
//   heads, so the causal-conv ring update is race-free without atomics or
//   extra launches:
//     q channels [kh*128, +128), k channels [2048+kh*128, +128),
//     v channels [4096+3kh*128, +384)  (v-heads 3kh..3kh+2)
//
//   Phase A: causal conv1d (kernel 4, BF16 weights, no bias) + silu over the
//            owned channels; ring shifted in place; conv output written back
//            into qkv_raw.
//   Phase B: per v-head wave: l2-normalize q,k; S *= exp(g);
//            kv = S^T k; delta = (v-kv)*beta; S += k x delta;
//            out = S^T (q/sqrt(128)). Fused into 3 passes over S
//            (decay+kv, update, out) instead of the old 5.
//   Phase C: RMSNormGated: out = w * rms(out) * sigmoid(z),
//            written as f32 straight into out6144 (no bf16 round trip --
//            the reference's intermediate bf16 store is incidental, and f32
//            is strictly more accurate for the following GEMV).
//
// Projections {qkv[10240], z[6144], a[48], b[48]} all read mixed[2560]:
// ONE fused multi-GEMV launch. Then out_proj GEMV. 3 launches total.
#include "qf8_dense.h"

__global__ void k8_gdn_fused(float *__restrict__ qkv,
                             const float *__restrict__ a_in,
                             const float *__restrict__ b_in,
                             const float *__restrict__ z,
                             const uint16_t *__restrict__ convw,
                             const float *__restrict__ A_log,
                             const float *__restrict__ dt_bias,
                             const uint16_t *__restrict__ gdn_norm,
                             float *__restrict__ ring,
                             float *__restrict__ S,
                             float *__restrict__ out6144) {
    const int kh = blockIdx.x;
    const int tid = threadIdx.x;              // 0..191
    const int wave = tid / QF8_WAVE;          // v-head slot 0..2
    const int lane = tid % QF8_WAVE;
    const int h = 3 * kh + wave;              // v-head

    // ---- Phase A: conv1d + silu + ring update for owned channels ----------
    for (int idx = tid; idx < 640; idx += 192) {
        int ch;
        if (idx < 128)      ch = kh * 128 + idx;
        else if (idx < 256) ch = 2048 + kh * 128 + (idx - 128);
        else                ch = 4096 + 3 * kh * 128 + (idx - 256);
        float cur = qkv[ch];
        float o = qf8_bf2f(convw[ch * 4 + 0]) * ring[0 * QF8_DINN + ch]
                + qf8_bf2f(convw[ch * 4 + 1]) * ring[1 * QF8_DINN + ch]
                + qf8_bf2f(convw[ch * 4 + 2]) * ring[2 * QF8_DINN + ch]
                + qf8_bf2f(convw[ch * 4 + 3]) * cur;
        ring[0 * QF8_DINN + ch] = ring[1 * QF8_DINN + ch];
        ring[1 * QF8_DINN + ch] = ring[2 * QF8_DINN + ch];
        ring[2 * QF8_DINN + ch] = cur;
        qkv[ch] = qf8_silu(o);
    }
    __syncthreads();

    // ---- Phase B: recurrence ----------------------------------------------
    __shared__ float ks[3][QF8_GDN_D], qs[3][QF8_GDN_D];
    float q0 = qkv[kh * 128 + lane],          q1 = qkv[kh * 128 + lane + 64];
    float k0 = qkv[2048 + kh * 128 + lane],   k1 = qkv[2048 + kh * 128 + lane + 64];
    float q2 = qf8_wave_sum(q0 * q0 + q1 * q1);
    float k2 = qf8_wave_sum(k0 * k0 + k1 * k1);
    q2 = __shfl_sync(QF8_WMASK, q2, 0);
    k2 = __shfl_sync(QF8_WMASK, k2, 0);
    float rq = rsqrtf(q2 + 1e-6f), rk = rsqrtf(k2 + 1e-6f);
    const float scale = 1.f / sqrtf((float)QF8_GDN_D);
    ks[wave][lane]      = k0 * rk;
    ks[wave][lane + 64] = k1 * rk;
    qs[wave][lane]      = q0 * rq * scale;
    qs[wave][lane + 64] = q1 * rq * scale;
    __syncthreads();

    float g = -__expf(A_log[h]) * log1pf(expf(a_in[h] + dt_bias[h]));
    float decay = __expf(g);
    float beta = qf8_sig(b_in[h]);

    float *Sh = S + (size_t)h * QF8_GDN_D * QF8_GDN_D;
    float o[2];
    #pragma unroll
    for (int cc = 0; cc < 2; cc++) {
        int col = lane + cc * QF8_WAVE;
        float v = qkv[4096 + h * QF8_GDN_D + col];
        // pass 1: decay + kv
        float kv = 0.f;
        for (int r = 0; r < QF8_GDN_D; r++) {
            float sv = Sh[r * QF8_GDN_D + col] * decay;
            Sh[r * QF8_GDN_D + col] = sv;
            kv += sv * ks[wave][r];
        }
        float d = (v - kv) * beta;
        // pass 2: rank-1 update
        for (int r = 0; r < QF8_GDN_D; r++)
            Sh[r * QF8_GDN_D + col] += ks[wave][r] * d;
        // pass 3: output
        float oo = 0.f;
        for (int r = 0; r < QF8_GDN_D; r++)
            oo += Sh[r * QF8_GDN_D + col] * qs[wave][r];
        o[cc] = oo;
    }

    // ---- Phase C: RMSNormGated --------------------------------------------
    // Gate activation is SIGMOID, not silu. config.json sets
    // "output_gate_type": "sigmoid", and Qwen4ExpTextGatedDeltaNet builds its
    // Qwen4ExpTextRMSNormGated with activation=config.output_gate_type, so the
    // reference computes weight * rmsnorm(v) * sigmoid(g).
    // Using silu(g) = g*sigmoid(g) multiplies every GDN output by an extra
    // factor of g - and GDN is 36 of the 48 production layers.
    float ss = qf8_wave_sum(o[0] * o[0] + o[1] * o[1]);
    ss = __shfl_sync(QF8_WMASK, ss, 0);
    float rs = rsqrtf(ss / QF8_GDN_D + 1e-6f);
    #pragma unroll
    for (int cc = 0; cc < 2; cc++) {
        int col = lane + cc * QF8_WAVE;
        float zv = z[h * QF8_GDN_D + col];
        out6144[h * QF8_GDN_D + col] =
            qf8_bf2f(gdn_norm[col]) * (o[cc] * rs) * qf8_sig(zv);
    }
}

void qf8_gdn_mixer(const Qf8GdnW *w, const float *mixed,
                   float *gdnS, float *convring,
                   float *qkv_raw, float *z6144, float *a48, float *b48,
                   float *out6144, float *y2560, hipStream_t s) {
    Qf8GemvMulti m;
    m.W[0] = w->qkv;   m.y[0] = qkv_raw; m.rows[0] = QF8_DINN;
    m.W[1] = w->zgate; m.y[1] = z6144;   m.rows[1] = 6144;
    m.W[2] = w->in_a;  m.y[2] = a48;     m.rows[2] = QF8_DTRANK;
    m.W[3] = w->in_b;  m.y[3] = b48;     m.rows[3] = QF8_DTRANK;
    m.n = 4;
    qf8_gemv_multi(&m, mixed, QF8_NEMBD, s);
    QF8_LAUNCH(k8_gdn_fused, QF8_GDN_KH, 192, 0, s,
               qkv_raw, a48, b48, z6144, w->conv1d, w->a_log, w->dt_bias,
               w->gdn_norm, convring, gdnS, out6144);
    qf8_gemv_bf16(w->gdn_out, out6144, y2560, QF8_NEMBD, 6144, 1.f, 0.f, s);
}

// ---- the same mixer as THREE separately callable steps ------------------
// (M=1 product path: the stage executor brackets each step with a section
// mark so the per-layer cost of projections / recurrence / out-projection is
// visible without a microbenchmark. Same launches, same order, same buffers.)
void qf8_gdn_proj(const Qf8GdnW *w, const float *mixed,
                  float *qkv_raw, float *z6144, float *a48, float *b48, hipStream_t s) {
    Qf8GemvMulti m;
    m.W[0] = w->qkv;   m.y[0] = qkv_raw; m.rows[0] = QF8_DINN;
    m.W[1] = w->zgate; m.y[1] = z6144;   m.rows[1] = 6144;
    m.W[2] = w->in_a;  m.y[2] = a48;     m.rows[2] = QF8_DTRANK;
    m.W[3] = w->in_b;  m.y[3] = b48;     m.rows[3] = QF8_DTRANK;
    m.n = 4;
    qf8_gemv_multi(&m, mixed, QF8_NEMBD, s);
}
void qf8_gdn_recur(const Qf8GdnW *w, float *gdnS, float *convring,
                   float *qkv_raw, const float *z6144, const float *a48, const float *b48,
                   float *out6144, hipStream_t s) {
    QF8_LAUNCH(k8_gdn_fused, QF8_GDN_KH, 192, 0, s,
               qkv_raw, a48, b48, z6144, w->conv1d, w->a_log, w->dt_bias,
               w->gdn_norm, convring, gdnS, out6144);
}
void qf8_gdn_outproj(const Qf8GdnW *w, const float *out6144, float *y2560, hipStream_t s) {
    qf8_gemv_bf16(w->gdn_out, out6144, y2560, QF8_NEMBD, 6144, 1.f, 0.f, s);
}
