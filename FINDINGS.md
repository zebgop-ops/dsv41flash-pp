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

## 5. Getting from 8 to 16.7 tok/s (same outputs)

- **Full CUDA graphs did nothing** (8.1 either way): GPU device time is only ~27 ms of the
  step (in-worker `torch.profiler` via `DSV41_DEBUG_PROFILE=1`; this image has no
  `/start_profile` route). Per-rank timing (`DSV41_DEBUG_SYNC=1`, `DSV41_DEBUG_TIMING=1`)
  showed the CPU experts and the gaps around them were the rest.
- **kt-kernel's AVX2 MXFP4 MoE is one thread per expert.** 5.7-5.8 ms per layer-step at 8,
  16, 24 and 32 threads, linear in batch size: ~3 GB/s of packed weights per thread, compute-
  bound in the nibble decode. `overlay/hybrid/cpu_moe.cpp` splits every expert's rows across
  all threads (static schedule for decode so each thread streams contiguous rows, dynamic for
  prefill), decodes E2M1 through a byte LUT with the ×2 folded into the E8M0 scale, and
  accumulates in fp32: 2.4 ms per layer-step, ~44 GB/s against a measured ~52 GB/s DRAM
  ceiling (`bw` test, 16 threads). Exact against the torch reference (cosine 0.999999, max
  abs 1e-4 on outputs of magnitude 3e-3). 12 CPU layers: 10.9 tok/s.
- **The Marlin MXFP4 repack needs raw + packed on the GPU at once.** `prepare_moe_mxfp4_layer_
  for_marlin` builds the packed w13/w2 while the raw tensors are still referenced by the layer
  and by two caller frames: ~6.7 GiB of transient per layer, which is what limited a rank to
  7 expert layers (OOM at 62.4 GiB with 8). `overlay/hybrid/marlin_staged.py` copies the raw
  tensor to host memory, drops the GPU copy, and packs expert by expert into the preallocated
  output; `patch_marlin_staged.py` stops the callers from holding references. Bit-exact
  against vLLM's own prepare (`test_marlin_staged.py`). Result: 8 expert layers per rank,
  partition `8,12,8,12`, 9 CPU layers, 16.7 tok/s.
- **Pinned staging is a trap.** The first staged boot used pinned host buffers; torch's
  pinned allocator caches freed blocks for the life of the process, so each worker kept
  ~6.7 GiB locked and the host OOM-killed worker 3. Pageable staging fixed it (peak 87 GB).
- **Triton autotune under full-graph capture.** With `FULL_AND_PIECEWISE` the sparse
  indexer's paged MQA-logits kernel is captured; one autotune key (block size 128, finalized
  after the indexer's warmup ran with the placeholder 16) was missing, and the benchmark's
  device sync invalidated the capture. The warmup now covers 128, and `mqa_logits_triton.py`
  installs an `Autotuner._bench` guard that takes the first config (and logs the key) if a
  benchmark is ever requested during capture.

## 6. Smaller upstream issues met on the way

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

## 7. Speculative decoding under PP4 with CPU experts (n-gram, same outputs on code)

**What it buys.** Prompt-lookup (n-gram) drafts cost nothing to produce and are verified in one
forward pass. On this box a verify step with 1+K tokens is *not* cheap, though: the CPU-expert
layers stream the weights of every expert any row routes to, so a 4-row step moves up to 4x the
expert bytes (measured ~2.5x the wall time of a 1-row step). Speculation therefore pays only where
drafts are accepted most of the time — code edits, rewrites, quoting context — and costs a little
elsewhere. Numbers in RESULTS.md.

**What was broken, in the order it was found.** Every item below reproduced as garbage, a crash, or
a silent divergence from the eager reference (`tools/lpcheck.py` compares top-5 logprobs at three
positions against a saved eager run; `tools/specbench.py` compares greedy outputs against the
no-spec run and reads acceptance from `/metrics`).

1. *Non-last ranks have no drafter.* The V1 runner reads `self.drafter` on every rank; with PP the
   attribute only exists on the last one. `patch_v1_spec_pp.py` defines it as `None` first.
2. *Token bookkeeping across ranks.* Under PP the scheduler ships sampled tokens back to the
   workers (there is no direct last→first rank path). It shipped only the one token scheduled this
   step, so accepted draft tokens never reached ranks 0-2; the per-request token arrays drifted, and
   the next step indexed past the end. `patch_pp_spec_tokens.py` sends every token a worker has not
   seen yet, and the worker appends exactly those.
3. *Scheduling ahead of the output.* With the PP batch queue the scheduler could schedule a request
   again while its previous step was still in flight; with drafts pending that produced a batch made
   of drafts only, negative logits indices and a device assert. The same patch skips a request whose
   tokens are all in flight (non-async scheduling only).
4. *PIECEWISE graphs were captured without attention metadata.* The model gates Engram hashing and
   the PP shadow-source inserts on `attn_metadata`; during piecewise capture it is `None`, so both
   were baked into the graph as no-ops and decode logits depended on boot history.
   `patch_engram_piecewise.py` hands the model static (query_start_loc, slot_mapping, block_table)
   buffers of the Engram group, and `pp_shadow.py` runs its update as a breakable-cudagraph eager
   segment.
5. *Padded rows reached the CPU experts.* Graph padding (1 real row in a 4-row graph) ran full
   expert passes for the padding and changed the accumulation grouping of the real row.
   `patch_pad_mask.py` publishes `ForwardContext.is_padding` every step; `cpu_experts.py` masks the
   padded rows' routing (prose decode 7.0 → 12.7 tok/s at that point). Prefill-shaped batches are
   kept out of piecewise graphs (`patch_prefill_eager.py`).
6. *Capture sizes are rounded to multiples of K+1.* vLLM rounds every CUDA-graph capture size up to
   a multiple of `1 + num_speculative_tokens` whenever decode uses FULL graphs, so with K=3 the
   sizes 1, 2, 3 disappear and a draft-less step (one real token) replays a 4-row graph. The
   dispatcher also knows only one "uniform decode" query length (K+1), so 1-token steps fell to
   PIECEWISE graphs, whose eager attention/indexer/Engram segments cost ~35 ms of host time per
   step here. `patch_full_q1.py` adds a second family of FULL decode graphs with query length 1
   (`num_tokens == num_reqs`), dispatches draft-less steps to it, and skips the rounding
   (`DSV41_FULL_Q1=0` restores upstream behaviour). Logprobs on 1-token steps are then identical
   to eager (0.000 on every probe).

**What is left, and why it is structural.** A draft-less step still costs ~65 ms against ~58 ms
without speculation (streaming, thinking off). vLLM turns on *async scheduling* by default and
turns it off for CPU n-gram speculation. Without it the engine is synchronous: schedule → RPC →
rank-0 host prep (~4.5 ms) → the four-rank GPU chain → sample → a blocking `take_draft_token_ids`
RPC → next step; with it, the sampled token reaches ranks 0-2 by GPU broadcast and rank 0's prep
and launch for the next step overlap the current one. The GPU n-gram drafter keeps async
scheduling, but PP + async asserts `[num_reqs, 1]` sampled ids and its invalid-draft trimming
runs on every rank from a buffer only the last rank fills — it crashes on rank 0 as shipped.
Plumbing that through PP would recover at most the ~3 ms engine round trip, because rank 0's
prep must wait for the drafts anyway, so it was not pursued. The cross-rank timeline that
established this is `DSV41_DEBUG_TRACE=1` + `tools/tracetl.py` (steps 40-44: worker entry,
receive posted, launched, sent, GPU done, plus the engine's RPC issue time).

**Numerics.** Greedy output of the code-edit task is byte-identical with and without speculation.
Free prose diverges after a few hundred characters: a 4-row verify step evaluates the row with
different kernel tilings than a 1-row step, and this stack's eager path itself is not
batch-invariant (`tools/batchinv.py`: concurrent vs sequential prefill differs by up to 2.2 nats
on tail tokens with the same top-1). The deviations sit inside that envelope; rejection sampling
keeps the accepted tokens exactly those the 1+K-row verify pass would have sampled.

## Diagnostic switches (all off by default)

| switch | effect |
|---|---|
| `DSV41_CG=NONE` | no CUDA graphs (eager), for tracebacks that point at the right kernel |
| `DSV41_EXTRA_DOCKER="-e CUDA_LAUNCH_BLOCKING=1 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=0"` | synchronous launches; the exception is logged before the NCCL watchdog aborts |
| `DSV41_DEBUG_STATS=1` | per-layer \|h\| / residual stats every step (each line syncs) |
| `DSV41_DEBUG_DUMP=/dump`, `DSV41_DEBUG_DUMP_LAYERS=0,1,2` | attn/ffn input+output dumps for 1 < T ≤ 64 batches, plus per-stage attention dumps for `ktests/ref_stages.py` |
| `DSV41_DEBUG_SYNC=1` | device syncs around the PP receive and per KV group in slot mapping; logs block-table geometry and per-step recv-wait/forward times |
| `DSV41_DEBUG_TIMING=1` | per-layer wall time (with syncs) every 16th decode step |
| `DSV41_DEBUG_PROFILE=1` | wraps the 41st decode step in `torch.profiler` and logs the top kernels by device time per rank |
| `DSV41_DEBUG_SPEC=1` | logs every small batch's CUDA-graph dispatch (tokens, requests, uniform, mode, descriptor) |
| `DSV41_DEBUG_ENGRAM=1` | checks the static Engram metadata buffers against the live attention metadata every step |
| `DSV41_DEBUG_GDUMP=/dump` | graph-safe per-layer / sub-block activation dumps (`cudaLaunchHostFunc` + pinned buffers), compared with `ktests/diff_gdump.py` |
| `DSV41_DEBUG_CORE=1` | engine-core phase means (schedule, RPC issue, wait, take_draft, update) every 32 real steps |
| `DSV41_DEBUG_TRACE=1` | cross-rank timeline of decode steps 40-44 (worker entry, receive posted, launched, sent, GPU done, engine issue) rendered by `tools/tracetl.py` |
| `DSV41_FULL_Q1=0` | upstream behaviour under speculation: capture sizes rounded to K+1, draft-less steps in PIECEWISE graphs |
| `DSV41_PREFILL_EAGER=0` | let prefill-shaped batches use piecewise graphs |
| `CPU_MOE_TRACE=1` | the native CPU kernel prints its phase times |
| `DSV41_ENGRAM_ZERO=1` | Engram lookups return zeros (isolates the SSD path) |
| `DSV41_MHC_TORCH=1` | torch mHC instead of tilelang |
