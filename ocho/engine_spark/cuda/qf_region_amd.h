// qf_region_amd.h - Spark-side client for the AMD dependency-closed region.
// The AMD box holds the whole model across its four MI50s; a region step runs
// its rows through ALL 48 layers there and returns sampled token ids. Only
// tokens cross the wire. Enables  AMD region(A) || Spark region(B).
//   QF_REGION_AMD=1, QF_REGION_AMD_HOST, QF_REGION_AMD_PORT (default 5578)
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
int qf_region_amd_enabled(void);
int qf_region_amd_reset(void);
// non-blocking beyond the send; pairs with _collect
int qf_region_amd_submit(long long pos, int M, const int *tokens, const long *posM);
int qf_region_amd_collect(int M, int *tokens_out);
#ifdef __cplusplus
}
#endif
