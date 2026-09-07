# FLA chunked gated delta rule - ahead-of-time Triton cubins (Spark prefill GDN)

Runtime: native only. cuda/qf_gdn_fla.cu loads these five cubins with the CUDA Driver API
(cuModuleLoadData / cuLaunchKernel) on the engine's stream; entry names, warps, shared memory and the
launch ABI are fixed in that source. No Python, torch, Triton, venv or pip is needed to start, load,
prefill, refill or decode.

Offline forge (build-time only, run once): harvest_recipe.py hooks Triton's JIT launch path while
running fla.ops.gated_delta_rule.chunk_gated_delta_rule on this GB10 (H=48, K=V=128, BT=64, bf16 q/k/v,
fp32 g/beta/state, initial_state given, final state stored) and writes each launched kernel's cubin plus
manifest.json (grid, params, constexprs, warps, shared). Produced 2026-09-06 with Triton 3.8.0 and
flash-linear-attention (MIT license, https://github.com/fla-org/flash-linear-attention) on CUDA 13.0
for sm_121. One cubin per kernel serves every chunk length we use (verified identical hashes for
T = 2048, 1893, 256, 125, 64, 32, 16).

Launch ABI (Triton 3.8): runtime params in signature order, then two unused scratch pointers.
  chunk_local_cumsum_scalar_kernel(s, o, scale, T)                       grid (NT, 48)   warps 1  smem 0
  chunk_gated_delta_rule_fwd_kkt_solve_kernel(k, g, beta, A, T)          grid (NT, 48)   warps 1  smem 9216
  recompute_w_u_fwd_kernel(k, v, beta, w, u, A, g, T)                    grid (NT, 48)   warps 8  smem 28672
  chunk_gated_delta_rule_fwd_kernel_h_blockdim64(k, v=u, w, v_new, g, h, h0, ht, T)  grid (4, 48) warps 2 smem 16384
  chunk_fwd_kernel_o(q, k, v=v_new, h, g, o, scale, T)                   grid (1, NT, 48) warps 8 smem 65536
