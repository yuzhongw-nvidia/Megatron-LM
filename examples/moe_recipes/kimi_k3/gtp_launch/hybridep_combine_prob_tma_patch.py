# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Patch a copy of DeepEP's hybrid_ep_backend.cuh: TMA copy for unaligned prob slices.

Target: DeepEP hybrid-ep @ 10d4dd7, ``inter_node_G2S_warp_group_device_function`` of the
combine kernel with ``BACKWARD_COMBINE``. For every (dst token, source rank) piece the
kernel copies the source rank's ``NUM_OF_EXPERTS_PER_RANK`` probability floats. When that
slice is not a multiple of 16 bytes the original code falls back to scalar loads from peer
memory in the single producer thread, which is latency-bound (measured 3.3x slower kernel
with 14 experts per rank).

The patch copies the enclosing 16-byte-aligned window with one TMA bulk copy instead. The
window never leaves the token's own prob row, because the row starts 16-byte aligned and
its size (experts per rank x ranks per node x 4 bytes) is a multiple of 16 bytes (both
already required by the kernel). The reduction warp group skips the leading floats of the
window, so the accumulated values are the same floats as before.

Only the inter-node G2S site is changed (the one used when all ranks share one NVLink
domain, NUM_OF_NODES == 1, and for local sources in multi-node builds). The intra-node G2S
site of multi-node builds and the dispatch-side store are left as they are.

Usage: hybridep_combine_prob_tma_patch.py <path/to/hybrid_ep_backend.cuh>
The file is modified in place; every anchor must match exactly once or the script fails
without writing.
"""

import hashlib
import sys

EXPECTED_SHA256 = "a3543798685c67412c3cb6d891a1ca2953b892327efcddcf4a3a9ef13f851842"

HELPER_ANCHOR = """template<int NUM_OF_FLOATS>
inline __device__ void scalar_copy_float_slice(float* dst, const float* src) {"""

HELPER = """// Floats reserved per G2S stage for one source rank's prob slice. TMA needs 16B-aligned
// addresses and sizes: a slice that is not a multiple of 16B is copied as its enclosing
// 16B-aligned window (at most 12 leading bytes), so the stage must hold that window.
template<int NUM_OF_FLOATS>
__host__ __device__ constexpr int prob_slice_g2s_slot_floats() {
  return ((NUM_OF_FLOATS * 4) >= 16 && (NUM_OF_FLOATS * 4) % 16 == 0)
             ? NUM_OF_FLOATS
             : (((NUM_OF_FLOATS * 4) + 12 + 15) / 16) * 4;
}

"""

STRUCT_OLD = (
    "  alignas(16) float inter_node_prob_G2S_buffer[NUM_OF_STAGES_G2S][NUM_OF_EXPERTS_PER_RANK];\n"
)
STRUCT_NEW = (
    "  alignas(16) float inter_node_prob_G2S_buffer[NUM_OF_STAGES_G2S]"
    "[prob_slice_g2s_slot_floats<NUM_OF_EXPERTS_PER_RANK>()];\n"
)

PRODUCER_ANCHOR = (
    "float* local_prob_addr = &smem_buffer_ptr->inter_node_prob_G2S_buffer[token_stage][0];"
)
PRODUCER_OLD = """                    } else {
                      scalar_copy_float_slice<NUM_OF_EXPERTS_PER_RANK>(local_prob_addr, remote_prob_addr);
                    }
"""
PRODUCER_NEW = """                    } else {
                      // The slice is not a multiple of 16B, so copy its enclosing 16B-aligned window with TMA.
                      // The window stays inside this token's prob row: the row starts 16B-aligned and its size
                      // is a multiple of 16B. The reduction warp group skips the leading bytes.
                      const uint32_t prob_lead_bytes =
                          (uint32_t)((current_src_token_id * NUM_OF_EXPERTS_PER_RANK * (int)sizeof(float)) % 16);
                      const uint32_t prob_window_bytes =
                          (prob_lead_bytes + (uint32_t)(NUM_OF_EXPERTS_PER_RANK * sizeof(float)) + 15u) & ~15u;
                      cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                               cuda::ptx::space_global,
                                               reinterpret_cast<void*>(local_prob_addr),
                                               reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(remote_prob_addr) - prob_lead_bytes),
                                               prob_window_bytes,
                                               &smem_buffer_ptr->inter_node_mbarrier_G2S_buffer[token_stage][0]);

                      total_tx_size += prob_window_bytes;
                    }
"""

CONSUMER_OLD = """              int src_rank = smem_buffer_ptr->inter_node_prob_src_rank_G2S_buffer[token_stage];
              int prob_offset = src_rank * NUM_OF_EXPERTS_PER_RANK;
              #pragma unroll
              for(int n = 0; n < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; n++){
                int global_element_id = thread_rank_within_pipeline + n * NUM_OF_THREADS_PER_PIPELINE;
                int local_element_id = global_element_id - prob_offset;
                if(local_element_id >= 0 && local_element_id < NUM_OF_EXPERTS_PER_RANK){
                  acc_prob[0][n] += load_prob_base_ptr[local_element_id];
                }
              }
"""
CONSUMER_NEW = """              int src_rank = smem_buffer_ptr->inter_node_prob_src_rank_G2S_buffer[token_stage];
              int prob_offset = src_rank * NUM_OF_EXPERTS_PER_RANK;
              // A slice that is not a multiple of 16B arrives as its enclosing 16B-aligned window:
              // skip the leading floats that belong to the previous rank's slice.
              int prob_lead_floats = 0;
              if constexpr(!((NUM_OF_EXPERTS_PER_RANK * sizeof(float)) >= 16 &&
                             (NUM_OF_EXPERTS_PER_RANK * sizeof(float)) % 16 == 0)){
                prob_lead_floats = ((src_rank * NUM_OF_EXPERTS_PER_RANK * (int)sizeof(float)) % 16) / (int)sizeof(float);
              }
              #pragma unroll
              for(int n = 0; n < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; n++){
                int global_element_id = thread_rank_within_pipeline + n * NUM_OF_THREADS_PER_PIPELINE;
                int local_element_id = global_element_id - prob_offset;
                if(local_element_id >= 0 && local_element_id < NUM_OF_EXPERTS_PER_RANK){
                  acc_prob[0][n] += load_prob_base_ptr[prob_lead_floats + local_element_id];
                }
              }
"""


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"patch anchor '{label}' matched {count} times, expected exactly 1")
    return text.replace(old, new)


def patch_text(text: str) -> str:
    text = _replace_once(text, HELPER_ANCHOR, HELPER + HELPER_ANCHOR, "helper")
    text = _replace_once(text, STRUCT_OLD, STRUCT_NEW, "single-node smem struct")

    # The scalar fallback text also exists in the intra-node G2S function; patch only the
    # occurrence that follows the inter-node producer anchor.
    if text.count(PRODUCER_ANCHOR) != 1:
        raise SystemExit("producer anchor did not match exactly once")
    start = text.index(PRODUCER_ANCHOR)
    pos = text.find(PRODUCER_OLD, start)
    if pos < 0 or pos - start > 2500:
        raise SystemExit(
            "scalar fallback block not found right after the inter-node producer anchor"
        )
    text = text[:pos] + PRODUCER_NEW + text[pos + len(PRODUCER_OLD) :]

    text = _replace_once(text, CONSUMER_OLD, CONSUMER_NEW, "inter-node reduction consumer")
    return text


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        original = f.read()
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
    print(f"input sha256 {digest}")
    if digest != EXPECTED_SHA256:
        print(
            f"WARNING: input differs from DeepEP 10d4dd7 (expected {EXPECTED_SHA256}); anchors decide"
        )
    patched = patch_text(original)
    with open(path, "w", encoding="utf-8") as f:
        f.write(patched)
    print(f"patched sha256 {hashlib.sha256(patched.encode('utf-8')).hexdigest()}")


if __name__ == "__main__":
    main()
