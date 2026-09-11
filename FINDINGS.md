# Findings

Root causes, in the order they cost boots (each boot is 7-8 minutes here: 475 GiB of shards,
12 layers of experts into kt-kernel, CUDA-graph capture). The boot-by-boot log is in
[PLAN.md](PLAN.md).

## 1. Garbage output with every kernel passing in isolation

**Symptom.** The server booted, every request completed, the text was garbage. Prompt
logprobs were wrong from position 1, so the per-token pipeline was broken, not context.
Every component passed on real weights when tested alone: dense MXFP8 Marlin, Marlin MXFP4
MoE (with the swiglu clamp), kt-kernel CPU experts, Engram rows, fp8 sm80 helpers, Triton
MQA logits, Triton sparse prefill/decode, the V4.1 cache quant round-trip, inverse RoPE,
the fused Q/KV RoPE insert op, hc_collapse, mHC tilelang vs torch. Engram-zero runs and
partitions without index-shadows were garbage too.

**Method.** A torch re-implementation of one block from DeepSeek's reference `model.py`,
fed with vLLM's own hooked inputs (`DSV41_DEBUG_DUMP`, `ktests/ref_layer0_v2.py`): MoE out
matched at cosine 0.99999, attention out at 0.92. Then per-stage dumps inside the Ampere
attention (`ampere_sparse.py`, `ktests/ref_stages.py`): q/kv projections 1.0000, RoPE insert
1.0000, gathered K 0.9998, sparse kernel 1.0000 given vLLM's own q/K, output projection 0.92
with mean |out| 0.00019 against 0.63 in the reference: the o_proj output was ~2^-11 too
small and block-wise distorted.

**Cause.** On SM8x the bmm-capable `wo_a` linear resolves to vLLM's MXFP8 *emulation*
kernel, which dequantizes the weight to bf16 at load but keeps the E8M0 `weight_scale`
parameter for its dtype asserts. The ROCm attention's `_get_cached_wo_a_bf16` sees a
`weight_scale` and multiplies the (already dequantized) weight by it again.

**Fix.** `DeepseekV41AmpereMLAAttention._o_proj` installs the bf16 weight as the cached einsum
operand directly (`wo_a._dsv4_wo_a_bf16`), so no scale is re-applied. Verified: o_proj cosine
1.0000, layer-1 attention 0.9999, correct chat answers.

## 2. Xid 31 MMU fault on rank 2 for prompts ≥ ~2k tokens

**Symptom.** 819-token prompts fine; 1,965 and 5,567 tokens killed worker 2 with `MMU Fault:
ENGINE GRAPHICS ... FAULT_PDE ACCESS_TYPE_VIRT_READ` at a 512 MB-aligned address, always on
the same GPU, always reported by the next Triton launch (`compute_slot_mapping`).

**Method.** `CUDA_LAUNCH_BLOCKING=1` and `TORCH_NCCL_ASYNC_ERROR_HANDLING=0` via the new
`DSV41_EXTRA_DOCKER` passthrough, so the traceback survives the NCCL watchdog; per-layer
stats (`DSV41_DEBUG_STATS=1`) to show rank 2 had not entered its forward; device syncs
before and after the PP receive and per KV-cache group around the slot-mapping kernel
(`DSV41_DEBUG_SYNC=1` in `gpu_worker.py`), plus a dump of every group's block-table
geometry; a standalone 4-GPU NCCL pipeline send/recv test up to 512 MB per hop (clean).

**Cause.** KV-cache group 5 is the compressor's `CircularBufferSpec` ring (block_size 8, one
block per request, table width padded to 16). The V1 model runner (which we must use: Engram
lookback ids are only handled there) runs the generic token→slot kernel on every non-Mamba
group; that kernel reads `block_table[req, pos // 8]`, i.e. up to 16k entries past a
16-entry row. On Hopper with the V2 runner this never happens (`slot_mapping_enabled = not
CircularBufferSpec`); on the V1 runner it is a silent out-of-bounds read until the address
leaves mapped memory. The compressor builder computes its own ring slots from the block
table, so the generic mapping was never used.

**Fix.** `overlay/patch_v1_circular.py`: `SlotMappingMode.NONE` for circular-buffer groups in
`may_reinitialize_input_batch`, mirroring the V2 runner. Verified at 1,965 / 5,567 / 29,067
prompt tokens.

## 3. PP cuts inside a kv-sharing group

V4.1 layers 3-7 read layer 2's compressed KV / indexer K caches, 9-13 read layer 8's, 15-19
layer 14's and 21-39 layer 20's, all through the forward context on the *same rank*. With
the decoder group (20-39) at 138 GiB, legal cuts alone cannot balance four 64 GB cards.
`overlay/hybrid/pp_shadow.py` re-instantiates the source layer's attention on the receiving
rank under the source's prefixes (so the KV manager allocates its caches there), ships the
source's post-norm attention input in `IntermediateTensors`, and replays kv_score →
compressor → insert_cache → indexer K (and, for index sources, the top-k). Layer 20's
candidate blocks travel the same way. Partition `7,7,10,16`: rank 1 shadows index source 2,
rank 3 shadows kv source 20.

## 4. Memory rules on 64 GB cards and a 123 GB host

- At most 7 GPU-expert layers per rank: 8 layers OOM'd in the Marlin MXFP4 repack
  (54 GiB of experts + ~4.5 GiB repack transient) at 62.4 GiB.
- kt-kernel keeps its own aligned copy of the FP4 weights and fp32-converted scales; the
  Python loader's tensors must be dropped after `load_weights` or each layer costs 13.4 GiB
  of RAM instead of 8.3 and the host swaps (the first request then hangs to the 300 s RPC
  timeout).
- `--max-num-batched-tokens 2048` bounds the Engram pinned staging, the kt-kernel prefill
  chunk and the sparse-prefill gather workspace (2.2 GiB for the ratio-1 layers).
- `DSV41_MEM_CAP_FRACTION=0.965` turns a near-top-of-card allocation into a clean OOM instead
  of an Xid-31 wedge that survives until reboot.

## 5. Smaller upstream issues met on the way

- **KV-cache tensor builder under PP** (`patch_kv_groups.py`): a projected group with no local
  layers keeps the global `UniformTypeKVCacheSpecs`, and the builder iterated that dict and
  allocated tensors for other ranks' compressor ring buffers (rank 3 died in KV allocation).
- **DeepGEMM gating** (`patch_mla_indexer.py`): `mla/indexer.py` gated on `has_deep_gemm()`
  (package importable) rather than `is_deep_gemm_supported()`, so the first request raised
  "Unsupported architecture" inside the metadata builder; the other ranks kept waiting on the
  PP receive, so the engine did not die and requests timed out instead.
- **kt-kernel under CUDA graphs**: only capture-registered batch sizes get persistent pinned
  I/O buffers; a graph captured with a temporary buffer replays against freed pinned memory
  (segfault in the worker thread). `install_cpu_experts` registers vLLM's capture sizes.
- **Weight iterator**: skipping the Engram tables and CPU layers' experts must happen inside
  the safetensors iterator (`patch_weight_iter.py`); filtering later still calls
  `get_tensor` on a 94 GiB table.
- **Pinned buffers under the CUDA default device**: `torch.empty(..., pin_memory=True)`
  without an explicit `device="cpu"` inside a worker lands on the GPU.

## Diagnostic switches (all off by default)

| switch | effect |
|---|---|
| `DSV41_CG=NONE` | no CUDA graphs (eager), for tracebacks that point at the right kernel |
| `DSV41_EXTRA_DOCKER="-e CUDA_LAUNCH_BLOCKING=1 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=0"` | synchronous launches; the exception is logged before the NCCL watchdog aborts |
| `DSV41_DEBUG_STATS=1` | per-layer \|h\| / residual stats every step (each line syncs) |
| `DSV41_DEBUG_DUMP=/dump`, `DSV41_DEBUG_DUMP_LAYERS=0,1,2` | attn/ffn input+output dumps for 1 < T ≤ 64 batches, plus per-stage attention dumps for `ktests/ref_stages.py` |
| `DSV41_DEBUG_SYNC=1` | device syncs around the PP receive and per KV group in slot mapping; logs block-table geometry |
| `DSV41_ENGRAM_ZERO=1` | Engram lookups return zeros (isolates the SSD path) |
| `DSV41_MHC_TORCH=1` | torch mHC instead of tilelang |
