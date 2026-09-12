# DeepSeek-V4.1-Flash on 4x CMP 170HX — plan (2026-09-10)

Goal: serve deepseek-ai/DeepSeek-V4.1-Flash (native FP8 dense / MXFP4 experts / FP8 Engram)
on this box, with the cold layers' experts computed in CPU RAM and the Engram tables on SSD.

## Facts that drive the design
- Checkpoint: 475 GiB, 48 shards. Backbone ~284 GiB (40 x 6.72 GiB MXFP4 experts + 0.12 GiB
  attention per layer, DSpark 6.5 GiB, embed/head 2.5 GiB, vision 0.8 GiB). Engram = 2 tables
  (layers 1 and 14), 384M rows x 256 FP8 + [rows,8] UE8M0 scales = 94.6 GiB each (shards 47/48).
- Box: 4 x CMP 170HX 64 GiB (sm_80, PCIe Gen2 x4, no P2P; keep PyTorch peak <= ~61.5 GiB/card),
  123 GB RAM (qwen38 currently holds ~70 GB of it), 1.8 TB free on the NVMe (990 PRO).
  => ~218 GiB of weights fit on GPUs after KV/activations => ~10 layers of experts (~67 GiB)
  must live in CPU RAM; Engram (189 GiB) cannot be in RAM => SSD + page cache.
- vLLM: model definitions merged (PR #56228), frontend merged (#56208), umbrella PR #56214
  (kernels, config/engram.py, engine args) open; official image vllm/vllm-openai:deepseekv41-flash-0909
  exists. Attention backends in the PR: FlashMLA (sm90/100), FlashInfer (sm120), ROCm/Triton.
  No Ampere path. V4 got its Ampere path from haosdent/vllm@f8ea5bb by subclassing the ROCm
  Triton sparse-MLA attention + fp8_sm80 LUT helpers (that is what runs DSv4 here today).
- PR engram: ParallelEngramEmbedding(cpu_offload=True) keeps the shard in pinned host memory
  and gathers over UVA. Not viable at 189 GiB on a 123 GB host.
- CPU experts: KT-Kernel (pip kt-kernel 0.7.0.post2, cp312) has a native AVX2 MXFP4 MoE
  backend (operators/avx2/mxfp4-moe.hpp) and an MXFP4SafeTensorLoader that reads the V4/V4.1
  layout `layers.N.ffn.experts.E.w{1,3,2}.{weight,scale}` straight from the HF shards.
  Zen3 (5955WX) = AVX2+FMA, no AVX-512.
- Reference NVMe Engram design: 0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000 (MIT):
  row_store.cpp = direct-mapped bounded RAM cache + O_DIRECT pread from the safetensors shard,
  called via cudaLaunchHostFunc so it works inside CUDA graphs.

## Stack decision: vLLM image + Python overlay (same pattern as glm53-run / qwen38-run)
1. Base: vllm/vllm-openai:deepseekv41-flash-0909 (check sm_80 in its arch list first).
2. Overlay A — Ampere attention for deepseek_v4_1: subclass DeepseekV41ROCMAiterMLAAttention
   (Triton sparse MLA prefill/decode, bf16 o_proj) as in f8ea5bb's ampere_sparse.py; port
   fp8_sm80.py (LUT decode / RNE encode) into every Triton kernel that converts fp8 on the
   V4.1 path (rocm_aiter_mla_sparse.py, v4_1/common/ops/cache_utils.py,
   fused_compress_quant_cache.py, indexer_k_store.py, engram lookup); Triton mqa-logits
   indexer fallback (mqa_logits_triton.py + sparse_attn_indexer.py hunks, + 0005a/0006
   torch top-k fallback and row chunking); indexer K cache FP8 (not MXFP4); gate
   `_select_dsv4_attn_cls` on capability.major == 8.
3. Overlay B — Engram on SSD: new storage mode for ParallelEngramEmbedding: hash ids -> host
   callback (cudaLaunchHostFunc) -> row_store (pread from shard 47/48 + bounded RAM cache)
   -> pinned rows/scales -> H2D -> dequant on GPU (torch fp8->bf16, e8m0 scale).
4. Overlay C — CPU experts for the cold layers: HybridMoE for the configured layer set:
   GPU router (select_experts) -> KT-Kernel KTMoEWrapper(method="MXFP4") for CPU experts
   (optionally a hot subset on GPU via gpu_experts_mask) -> shared expert on GPU. vLLM must
   skip loading those layers' expert weights to GPU.
5. Overlay D — PP4 with cache sharing across ranks: V4.1 consumers read the compressed KV /
   indexer K caches of their kv-source layer (2, 8, 14, 20) through the forward context on the
   SAME rank ("PP splits inside a kv-sharing group are not supported"). The decoder group
   (layers 20-39, 138 GiB) cannot sit on one card, so a receiving rank gets a shadow copy of
   the source layer's compressor(+indexer K store) fed by the source layer's attention input
   shipped in IntermediateTensors, plus the candidate block buffer of layer 20.
6. DSpark spec-decode later (upstream now has broadcast_drafts under PP).

## Blockers needing the user
- Host NVIDIA userland is 610.57.04 but the loaded kernel module is 610.43.02
  (/var/run/reboot-required exists): every NEW CUDA container fails to init. Reboot needed.
- All four GPUs and ~70 GB RAM are held by qwen38-pp (Up 4 days). It has to be stopped.

## Status log
- 2026-09-10 15:20 download started: hf download deepseek-ai/DeepSeek-V4.1-Flash (log dl-v41.log)
- 2026-09-10 15:50 docker pull vllm/vllm-openai:deepseekv41-flash-0909 (log pull-image.log)

## PP4 layout (decided 2026-09-10, after reading attention.py/compressor.py)
Consumers find their kv-source's compressed-KV cache and indexer K cache through
`static_forward_context[<source attn prefix>]` on the same rank. Index sources at
24/28/32/36 do not own K (they share kv source 20's K cache) and later indexers mask with
the candidate blocks published by layer 20.
=> Legal cut points without shadows: 2, 8, 14, 20. With a "shadow source" on the receiving
rank: any index-source boundary (24, 28, 32, 36).
Shadow source = the kv-source layer's attention module instantiated again on the next rank
(0.13 GiB), registered under the source's prefixes so the KV-cache manager allocates its
compressed KV / compressor state / indexer K caches there, and driven each step by the
source layer's post-norm attention input shipped in IntermediateTensors (bf16 [T,5120]):
kv_score GEMM -> compressor.forward -> insert_cache -> indexer._produce_k. The candidate
block buffer of layer 20 ([T,2048] int32) is shipped the same way.
Partition: R0 0-7 (8 layers, 54 GiB + embed) | R1 8-13 (40 GiB) | R2 14-23 (10 layers; experts
of 22-23 on CPU -> 54 GiB) | R3 24-39 (16 layers; experts of 8 of them on CPU -> 54 GiB,
+ lm_head 1.2 + DSpark 6.5 later). CPU experts total 10 layers = 67 GiB.

## Engram SSD backend — implemented (overlay/engram_ssd)
row_store.cpp (thread pool, pread through the page cache) + engram_ssd.py
(cudaLaunchHostFunc callback, pinned staging, torch fp8/e8m0 dequant). CPU test:
exact rows, 4.2 M rows/s warm.

## CPU experts — kt-kernel 0.7.0.post2 imports under the image's torch 2.13
(needs gguf/typer/rich; ext exposes AVX2MXFP4_MOE, selected automatically on this Zen3).
- kt-kernel AVX2 MXFP4 (CPU-only test, 16 threads, while qwen38 was serving): correct vs torch
  reference (rel err 0.3%); V4.1 shape E=384/H=5120/I=2304/top-6: M=1 5.35 ms per layer-step,
  M=4 15.4 ms, M=32 125 ms (~19-26 GiB/s weight traffic; 32 threads is slower than 16).
  => 10 CPU layers cost ~55 ms/token at batch 1 -> decode ceiling ~15-18 tok/s from the CPU
  side; DSpark verification batches (M=6) cost ~20 ms/layer-step, so spec decode helps.

## Ampere overlay — work items (status 2026-09-10 evening)
- [x] overlay/vllm/models/deepseek_v4_1/ampere/ampere_sparse.py (subclass of the PR's ROCm Triton attention)
- [x] patch_select_attn.py (SM8x -> Ampere class), patch_registry.py (TRITON_MLA_SPARSE_DSV41 / TRITON_SPARSE_SWA_DSV41)
- [x] patch_v41_ops_fp8.py (cache_utils / fused_compress_quant_cache / indexer_k_store encode+decode via fp8_sm80)
- [x] fp8_sm80.py + mqa_logits_triton.py staged into overlay/vllm/v1/attention/ops/
- [x] rocm_aiter_mla_sparse.py: patch_rocm_sparse_lut.py (regex port of the f8ea5bb hunks; dry-run on main OK)
- [x] sparse_attn_indexer.py: patch_indexer.py (Triton fallback, torch top-k, row chunking default 64, persistent_topk SM90 gate; dry-run on main OK). NOTE: V4.1 attention.py must pass num_heads=self.n_head to SparseAttnIndexer for the autotune warmup
- [x] fused_indexer_q.py: patch_fused_indexer_q.py (software encode, uint8 warmup/launch pointer, CuTe DSL gate) -- dry-run on main OK
- [x] CuTe DSL gates are local helpers in the patched files (main has no is_cutedsl_supported); patch_misc_sm80.py = tilelang prenorm torch fallback + execute_in_parallel capture guard (both still missing on main)
- [x] wo_a: Ampere class keeps is_bmm=True -> init_mxfp8_linear_kernel(bmm) picks EmulationMxfp8LinearKernel below SM90 (bf16 dequant at load, +0.09 GiB/layer); _o_proj drops any leftover weight_scale_inv so the bf16 einsum cache is exact. No Marlin exemption needed.
- [ ] engram.py: patch_engram.py (SSD storage) -- written, validated on the PR-head copy
- [x] PP shadow sources: overlay/hybrid/pp_shadow.py + patch_pp_shadow.py (plan per rank from get_pp_indices, ShadowSource under the source prefix, shipped attention inputs + candidate blocks in IntermediateTensors, loader remap) -- drafted, untested
- [x] CPU experts: overlay/hybrid/cpu_experts.py + patch_cpu_experts.py (meta-device routed experts for DSV41_CPU_EXPERT_LAYERS, loader skip, MoERunner._forward_impl override, kt-kernel load after GPU load) -- drafted, untested
- [ ] DSpark under PP (upstream has broadcast_drafts now; verify in image)

## Image facts (vllm/vllm-openai:deepseekv41-flash-0909, inspected 2026-09-10 16:15)
- vllm 0.1.dev20904+g179dd0fa9, Python 3.12, torch 2.13 cu130, triton 3.7.1, tilelang 0.1.12, flashinfer 0.6.18,
  cutlass-dsl 4.6.2; NO deep_gemm (Triton indexer fallback is mandatory anyway).
- Compiled arch list includes sm_80 (_C_stable_libtorch + _moe_C); the fused V4 KV-insert kernel is in the .so.
- --engram-config wired (EngramConfig.cpu_offload); image engram.py has DP head-sharding (patch re-anchored).
- Engram lookback ids are handled by the V1 runner only (v1/worker/gpu_model_runner.py); V2 runner
  (default here) has no engram support -> launcher sets VLLM_USE_V2_MODEL_RUNNER=0.
- No broadcast_drafts in v1/worker/gpu/pp_utils.py -> DSpark under PP unsupported in this image; SPEC off.
- Overlay applies cleanly: 43 files (overlay/vllm), all parse.

## Boot log (2026-09-10 evening, GPUs free after reboot + user stopped qwen38)
- boot 1: pinned staging buffers created under the cuda default-device context (fixed: explicit cpu device);
  rank 1 loader died in safetensors get_tensor (now skipped via DSV41_SKIP_WEIGHT_RE for engram tables +
  CPU-layer experts); rank 3 built consumers before the shadow source (fixed: shadow before make_layers,
  remap adds `.attn`). Stray root-owned deep_gemm in kt/site removed.
- boot 2: kt-kernel loaded CPU experts fine (ranks 2/3); rank 0 OOM in Marlin MXFP4 repack at 62.4 GiB with
  8 GPU layers (54 GiB experts + ~4.5 GiB repack transient). Rule: <= 7 GPU expert layers per rank.
  CPU set now 7,21-23,31-39 (13 layers, ~87 GiB RAM), util 0.92.
- boot 3: rank 0 loaded in 97 s at 51.67 GiB; MoERunner read quant_method.skip_forward_padding on the CPU-expert
  stub -> stub now carries a quant-method proxy (attributes delegated, no-op post-load).
- boot 4: SharedExperts wrapper only computes for its kernel's order -> CPU path runs the shared MLP directly.
- boot 5: profiling forward passed on all ranks (CPU experts, shadow source, Engram-SSD all executed);
  KV sized to 334,778 tokens at 131k (2.44 GiB on rank 0). Rank 3 died allocating KV: upstream bug --
  a projected KV group with no local layers keeps the global UniformTypeKVCacheSpecs dict and the tensor
  builder iterated it (compressor ring buffers of layers 2/8/14). patch_kv_groups.py filters by the
  group's own layer names. Instrumented v1/worker/utils.py names orphaned layers.
- boot 6: **server reached "Application startup complete"** (GPUs 55.8/46.5/55.5/58.8 GiB, KV 334,778 tokens,
  piecewise graphs captured), but the first request hung until the 300 s sample_tokens RPC timeout. Host RAM was
  124 GB used + 6.8 GB swap: kt-kernel keeps TWO host copies per CPU layer (Python loader tensors + C++ aligned
  buffers) = 13.4 GiB x 13 layers. Fix: cpu_experts.load_weights drops the Python-side lists after load.
  --max-num-batched-tokens lowered to 2048.
- boot 7: python copies freed; host RSS = 8.3 GB per CPU layer (fp4 + fp32-converted scales) + ~1.5 GB per
  worker; 13 layers still swapped (121 GB used). Shadow mechanism generalized to INDEX sources (the shadow
  replays the source's indexer top-k), so cuts are no longer restricted -> partition 7,7,10,16, CPU layers
  21-23,31-39 (12 layers, ~100 GB). Rank 1 shadows layer 2 (top-k), rank 3 shadows layer 20 (caches only).
- boot 8 (7,7,10,16): up, KV 749,878 tokens (5.7x @131k), RAM 112 GB used / 14 GB available. First request
  hung: rank 0 raised in the DSA indexer metadata builder -> DeepGEMM get_paged_mqa_logits_metadata
  ("Unsupported architecture"): mla/indexer.py gated on has_deep_gemm() (package present in the image);
  patch_mla_indexer.py switches all 4 sites to is_deep_gemm_supported(). The engine did not die on the
  worker exception (other ranks waited on the PP recv) -> requests time out at 300 s instead.
- boot 10 (graphs off): **first completed request** (HTTP 200, 32 tok in 13.6 s) but text is garbage.
  Bisection so far (all on GPU 0, real weights): dense MXFP8 Marlin exact (0.14%), Marlin MXFP4 MoE exact
  (0.6%, clamp ok), kt-kernel CPU experts exact incl. swiglu clamp, Engram SSD rows exact + graph replay,
  fp8 sm80 helpers bit-exact, Triton MQA logits + sparse decode/prefill kernels pass (V4 tests, with the
  V4 cache module patched for the tests), V4.1 cache quantize/dequantize round-trip ok, mHC tilelang
  (delayed pre / post / broadcast) exact vs torch. Engram-zero run: still garbage. No-index-shadow
  partition (8,6,10,16): still garbage. Raw /v1/completions garbage too (tokenizer sane).
  Remaining suspects: V4.1 attention/indexer *integration* on the Triton path, kv-shadow (rank 3),
  kt integration inside the MoE runner. Tools added: DSV41_DEBUG_STATS=1 (per-layer |h| stats),
  DSV41_MHC_TORCH=1 (torch mHC), DSV41_ENGRAM_ZERO=1.
- Further bisection (all pass): inverse RoPE Triton vs rotary native (needs a set_current_vllm_config
  fixture), CUDA fused Q/KV RoPE+UE8M0 insert op round-trip (Q inverse exact-ish, K dequant matches
  native RoPE), hc_collapse Triton, mHC tilelang vs torch, sparse decode/prefill Triton kernels.
  Per-layer |h| stats: bounded, no NaN, smooth growth. Prompt logprobs are wrong from position 1
  (after BOS alone) => per-token pipeline is wrong, not context attention.
  Next: torch oracle for block 0 (ktests/ref_layer0.py, built from DeepSeek's inference/model.py
  semantics) vs vLLM activation dumps (DSV41_DEBUG_DUMP=/dump; only short real batches).
- 2026-09-10 22:30 PT — ROOT CAUSE of garbage output (stage oracle ktests/ref_stages.py on
  DSV41_DEBUG_DUMP stage dumps from ampere_sparse.py): every attention stage matches the torch
  reference (q/kv projections+norms cos 1.0000, fused RoPE insert 1.0000, gathered fp8 K rows
  0.9998, Triton sparse prefill kernel vs attend(vLLM q, vLLM K) 1.0000) except the o_proj:
  the MXFP8 *emulation* kernel dequantizes wo_a to bf16 at load but leaves `weight_scale` on
  the layer, and the ROCm `_get_cached_wo_a_bf16` multiplies the bf16 weight by that E8M0 scale
  a second time -> o_proj output ~2^-11 too small and block-wise distorted (cos 0.92, mean
  |out| 0.00019 vs 0.63). Fix: DeepseekV41AmpereMLAAttention._o_proj installs
  `wo_a._dsv4_wo_a_bf16` from the bf16 weight directly (no scale). Re-verifying.
- 2026-09-10 23:30 PT — wo_a fix verified: layer-0 o_proj cos 1.0000, layer-1 attention cos 0.9999, chat answers
  correct (391, Canberra, Paris). Second bug (prompts >= ~2k tokens crash rank 2 with Xid 31 MMU fault):
  bracketed with DSV41_DEBUG_SYNC=1 (per-group slot-mapping syncs in gpu_worker.py) -> the generic
  token->slot kernel faults on kv-cache group 5 = the compressor CircularBufferSpec ring (block_size 8,
  1 block/req, table padded to 16 cols): it reads block_table[req, pos//8] far past the row. The V2 runner
  disables slot mapping for CircularBufferSpec; the V1 runner (needed for Engram lookback) does not ->
  new overlay/patch_v1_circular.py on v1/worker/gpu_model_runner.py (SlotMappingMode.NONE for ring groups).
  NCCL standalone pipeline send/recv (up to 512 MB/hop) is fine; launcher gained DSV41_EXTRA_DOCKER.
- 2026-09-10 23:45 PT — WORKING with CUDA graphs (PIECEWISE): plen 1965 tokens OK ("240" correct), smoke OK
  (391 / Canberra), needle 5.5k RECALL OK; decode 8.1 tok/s (graphs) vs ~5 eager; prefill ~62 tok/s at 5.5k.
  kt capture-size registration held up under graph replay (no repeat of the worker-2 segfault).
  Open: CPU thread tuning (DSV41_CPU_EXPERT_THREADS), DSpark spec decode, 30k needle (running), perf.
- 2026-09-10 23:55 PT — 30k needle RECALL OK (29,067 prompt tokens, 436 s, ~67 tok/s prefill). Bench C=4:
  15.8 tok/s aggregate (3.9/stream); single stream ~8 tok/s. DSpark stays off (no broadcast_drafts under PP
  in this image). Chat template has no thinking toggle; the model always emits reasoning (message.reasoning).
- 2026-09-11 09:25 PT — /goal 16 tok/s. Findings: FULL_AND_PIECEWISE graphs = no gain (8.1); kt-kernel's AVX2
  MXFP4 path is ~5.7 ms/layer-step regardless of threads (one thread per expert) -> replaced by
  overlay/hybrid/cpu_moe.cpp (AVX2/OpenMP, exact vs torch, 2.4 ms/layer-step = 44 GB/s of a ~52 GB/s DRAM
  ceiling; prefill 1.8 s per 2048-token chunk) -> 10.9 tok/s with 12 CPU layers. GPU device time ~27 ms/step
  (in-worker torch profiler, DSV41_DEBUG_PROFILE=1; nothing pathological). Load-time HBM transient of the Marlin
  MXFP4 repack (~6.7 GiB: raw+packed, refs pinned by caller frames) removed by staging the repack through host
  RAM (overlay/hybrid/marlin_staged.py, bit-exact vs vLLM's prepare) -> 8 GPU-expert layers per rank fit:
  partition 8,12,8,12, CPU layers 16-19,35-39 (9), util 0.97, KV 933k tokens. Pinned staging must be pageable
  (torch caches pinned blocks -> host OOM). Autotune guard for Triton under capture (mqa_logits_triton.py) and
  warmup for block 128. Worker debug hooks: DSV41_DEBUG_SYNC/TIMING/PROFILE.
- 2026-09-11 09:50 PT — GOAL MET: steady-state decode 16.6-16.9 tok/s (engine windows; 600-token completion
  36.4 s wall incl. prefill), 24.3 tok/s aggregate @4 streams, prefill ~100 tok/s; needle 5.5k and 29k RECALL OK;
  smoke answers unchanged. Launcher defaults updated (8,12,8,12 / 16-19,35-39 / util 0.97 / FULL_AND_PIECEWISE).
  Repo zebgop-ops/dsv41flash-pp updated (commit 17128a6). servers-top.py comment refreshed.
- 2026-09-11 15:30 PT — /goal spec decode. n-gram spec under PP: V1 runner crashed (drafter missing on non-last
  ranks -> patch_v1_spec_pp.py) then IndexError/corruption in PP token bookkeeping (scheduler sends only the
  scheduled non-draft token; worker's num_new_tokens drifts) -> patch_pp_spec_tokens.py (scheduler tracks
  per-request sent position, worker appends exactly what it gets). Prose acceptance 15% -> 7.9 tok/s (loss);
  outputs diverged from no-spec at the first draft step. Root cause found via graph-safe per-layer dumps
  (DSV41_DEBUG_GDUMP, cudaLaunchHostFunc callbacks that fire on graph replay): layer 0 replay == eager,
  layer 1 (first Engram layer) diverges. The model computes Engram hashes only when attn_metadata is a dict;
  PIECEWISE capture runs with attn_metadata=None, so piecewise graphs (small prefills, mixed and non-uniform
  spec batches) contain NO Engram injection -> boot/state-dependent logits (~1-2 nats), "Paris" flips.
  FULL decode graphs capture with real metadata (Engram present). Fix candidates: FULL_DECODE_ONLY (prefill
  eager) now; static-metadata Engram hashing under piecewise capture later.
- 2026-09-11 18:30 PT — spec decode, root causes and fixes so far (all in overlay/): (1) V1 runner: non-last PP
  ranks lacked `drafter` (patch_v1_spec_pp); (2) PP token bookkeeping: scheduler now sends every unseen token
  (patch_pp_spec_tokens), worker appends exactly; (3) PP batch queue scheduled a request again while in flight
  -> drafts-only steps, negative logits indices (scheduler guard in the same patch); (4) PIECEWISE graphs were
  captured with attn_metadata=None -> no Engram hashing (patch_engram_piecewise: static qsl/slot/block-table
  buffers from the runner) and the PP shadow-source update was baked in as a no-op (pp_shadow.run is now a
  breakable-cudagraph eager segment); (5) padded rows reached the CPU experts (4x cost, changed accumulation
  grouping) -> ForwardContext.is_padding published per step (patch_pad_mask) and masked in cpu_experts.py;
  (6) prefill-shaped batches run eagerly instead of piecewise (patch_prefill_eager). Verification: with
  FULL_DECODE_ONLY, logprobs == eager (0.000) on all probes; PIECEWISE decode == eager after (4)+(shadow);
  eager itself is NOT batch-invariant (concurrent vs sequential prefill: up to 2.2 nats on tail tokens, top-1
  same), so remaining ~0.5-nat deviations under spec (padded 4-row graphs) are batch-shape numerics.
  n-gram k=3: code-edit 97% draft acceptance (2.9/step), output identical to no-spec; prose acceptance 40-50%
  with min 3, decode 12.7 tok/s vs 16.3 no-spec (padding + drafts) -> trying explicit capture sizes 1..4 and
  prompt_lookup_min 5.
- 2026-09-11 18:05 PT — capture sizes [1,2,3,4,8,16] + FULL_AND_PIECEWISE: vLLM's adjust_cudagraph_sizes_for_spec_decode
  rounds every size up to a multiple of (K+1)=4 whenever decode_mode == FULL -> only 4/8/16 survive; prose (n-gram
  min 5, almost no drafts) 11.6 tok/s = pure padding cost (1 real row -> 4). code-edit 14.8 tok/s, output identical.
  Next: PIECEWISE (no rounding) with sizes 1..4 so no-draft steps run a 1-row graph.
- 2026-09-11 19:50 PT — q1 FULL graphs (overlay/patch_full_q1.py, DSV41_FULL_Q1=1): the dispatcher gets a second
  family of FULL decode graphs with query length 1 (num_tokens == num_reqs) and the K+1 rounding of capture sizes
  is skipped; draft-less steps now replay a 1-row FULL graph (DISPATCH log: tokens=1 -> FULL num_reqs=1 uniform)
  and lpcheck == eager (0.000) on all probes. Prose still ~12.5 tok/s in specbench (thinking on) vs 16.3 no-spec.
  Instrumentation added: DSV41_DEBUG_CORE=1 (engine-core phase means; overlay/vllm/v1/engine/core.py),
  DSV41_DEBUG_TRACE=1 (cross-rank timeline of steps 40-44: worker in/recv_posted/launched/sent/gpu_done +
  core exec_issue; tools/tracetl.py), DSV41_DEBUG_SYNC step lines now count only T=1 steps, profiler prints a
  self-CPU table (tools/profcpu.sh). Findings: with spec the engine core blocks in take_draft_token_ids for the
  whole step (harmless for 1 request) and issues 3 empty execute_model RPCs/step (batch queue depth 4 vs 1);
  numba thread pool is not the cause (NUMBA_NUM_THREADS=1 no change); per-kernel device times equal on rank 3,
  ~20% slower on ranks 0/1 in one profile (clock/contention noise?). Spec-on trace: rank1 GPU segment 23-29 ms
  (expected ~16 = 6 GPU + 9.6 CPU MoE), rank3 21-33 ms (expected ~20) -> CPU-expert host functions look
  contended; box has gnome-shell at 300-400% CPU during generation. Streaming (thinking off) spec-on: 61-70
  ms/step, 400 tok in 26.5-27.2 s. Next: same trace/streaming on no-spec for an apples-to-apples comparison.
- 2026-09-11 20:10 PT — ROOT CAUSE of the draft-less-step gap: vLLM enables *async scheduling* by default
  (config/vllm.py) and disables it for CPU "ngram" speculation. No-spec PP therefore overlaps the engine round
  trip + rank-0 host prep (~4.5 ms) + graph launch with the previous step (sampled ids reach ranks 0-2 by GPU
  broadcast, `_pp_broadcast_prev_sampled_token_ids`); with ngram spec the engine is synchronous: schedule ->
  exec RPC -> rank-0 prep -> ... -> sample -> take_draft RPC -> next. Trace (DSV41_DEBUG_TRACE): spec-on 1-token
  step ~65 ms = rank0 prep 4.5 + GPU chain 57 (rank0 6, rank1 23, rank2 4, rank3 24) + engine ~3; no-spec ~58.
  Streaming (thinking off): spec 61-70 ms/chunk vs no-spec 58-60. The rest of specbench's prose gap is failed
  draft steps: a 4-token verify step costs ~2.5x a 1-token step (CPU experts: up to 4x expert traffic per layer;
  code-edit 107 steps in ~17.7 s). ngram_gpu would keep async scheduling but PP+async asserts sampled ids
  [num_reqs,1] and the invalid-draft trimming needs a cross-rank broadcast -> not pursued (gain <= ~3 ms).
  Decision: keep CPU ngram; pick K and prompt_lookup_min by measurement (K=2 vs 3, min 5).
- 2026-09-11 20:45 PT — K sweep (prompt_lookup_min 5, max 8, q1 FULL graphs, sizes 1..4 + multiples of K+1):
  K=3: lpcheck 0.000; specbench prose 14.1 tok/s (5 draft steps, 80% acc) vs no-spec 16.3-16.8; code-edit
  15.3 tok/s (97% acc, 2.91/step) vs 12.4-12.7, output identical; streaming prose (thinking off) 26.1/27.9 s
  vs 21.8/25.2 no-spec; streaming code-edit 23.3 s (129 steps / 400 tokens). Launcher now derives CG mode
  (FULL_AND_PIECEWISE when DSV41_SPEC>0, else FULL_DECODE_ONLY) and capture sizes from DSV41_SPEC/DSV41_SEQS
  (DSV41_CG_SIZES overrides); ngram_gpu dropped (crashes on rank 0: token_ids_gpu_tensor missing; PP+async
  unsupported upstream). Repo synced (new patches, tools, FINDINGS §7, README); RESULTS pending K=2.
- 2026-09-11 21:05 PT — GOAL (spec decode) DONE, launcher defaults: DSV41_SPEC=3, ngram min 5 / max 8,
  FULL_AND_PIECEWISE with capture sizes [1,2,3,4,8,12,16]. Final run with defaults: lpcheck 0.000 on all probes;
  code-edit 15.1 tok/s incl. prefill (97% accepted, 2.91/step, greedy output identical to no-spec), streaming
  code 23.5 s / 400 tokens (129 steps); prose 13.6-14.3 tok/s (vs 16.3-16.8 no-spec), streaming 26.0/27.9 s.
  K=2 tied on prose (14.3) and lost on code (14.3 vs 15.3). DSV41_SPEC=0 restores the no-spec path unchanged.
  Repo dsv41flash-pp updated (patches, tools, FINDINGS §7, RESULTS spec table, README); servers-top/web notes
  updated (servers-web.service restart is the user's call).
- 2026-09-11 22:55 PT — DSPARK on the V1 runner under PP4 WORKS (overlay/hybrid/dspark_proposer.py +
  patch_dspark_v1.py; launcher DSV41_SPEC_METHOD=dspark -> K=5, CPU layers 16-19,33-39 so the 7.4 GiB draft +
  1 GiB embedding fit on rank 3). Port: DFlashProposer subclass with the anchor-first N-query layout, sequential
  Markov sampling (greedy), per-kv-group slot mappings/metadata (the hybrid manager puts each draft SWA cache in
  its own group), mtp.*-only checkpoint read (weight_utils _DSV41_ONLY_RE), embed.weight loaded from shard 2 on
  rank 3, lm_head aliased, draft_parallel_config pp=1, V1 "unsupported: dspark" check bypassed, async scheduling
  off under PP (the [num_reqs,1] broadcast assert), eagle-family runner gates guarded for ranks without a drafter,
  DFlash's non-causal metadata assert dropped (sparse-SWA handles causal=False internally). First numbers:
  lpcheck text same (0.5-0.9 nats at later positions = 6-row numerics); code-edit 98% accepted (4.91/step) but
  14.0 tok/s incl. prefill; prose 26% accepted (1.38/step, pos0 68% then 36/20/10/4%) 8.3 tok/s: a DSpark step
  costs ~265 ms (n-gram 6-row step 140 ms). Next: phase timing of the draft, confidence-based adaptive
  truncation (variable-length drafts through the sync scheduler path), CUDA graphs for the draft.
- 2026-09-11 23:30 PT — DSpark tuning. Draft cost is ~10 ms/step (inputs+meta 4.9, ctx insert 0.7, forward 1.9
  in a piecewise graph, Markov sampling 2.3). Confidence-based adaptive truncation added (cumulative confidence
  >= tau, variable-length drafts through the synchronous scheduler path; /dump/dspark_conf overrides
  DSV41_DSPARK_CONF at runtime) and FULL graph families for every query length 1..K (patch_full_qall; capture
  sizes 1,2,3,4,5,6,8,10,12,18,24 -> 20 FULL graphs). 300-token streaming, thinking off:
  prose tau 0/0.3/0.5/0.7: 47.9(400tok)/23.4/25.5/22.2 s (no-spec ~18 s, n-gram ~20 s);
  code-edit (copy from prompt): 20.5-24.3 s incl. ~9.5 s prefill (54 steps, 5.56 tok/step);
  FRESH code (LRU cache + tests, nothing to copy): 300 tokens in 15.1-15.3 s, 57-59 steps, 85% of drafts
  accepted = ~20 tok/s vs 16.7 no-spec / ~16 n-gram. A 6-row DSpark verify step costs ~260 ms vs ~140 ms for the
  n-gram K=5 shape (2 extra CPU layers ~+24 ms, draft +10 ms; ~85 ms unexplained -> tracing).
- 2026-09-11 23:55 PT — DSpark kept as an opt-in mode (DSV41_SPEC_METHOD=dspark, conf 0.7): fresh code +20%
  (~20 tok/s), copy-heavy code ~= n-gram, prose -15%. Cross-rank trace of a 6-row step: rank1 ~95 ms, rank3
  ~155 ms = CPU-expert layers at ~20 ms each (≈40 distinct experts/layer, DRAM-bound); moving the draft experts
  to CPU to free a target layer nets zero. Production rebooted on n-gram defaults and re-verified (lpcheck 0.000,
  code-edit identical, 14.8 tok/s incl. prefill; capture sizes now 1,2,3,4,6,8,12,16 with q=1..3 FULL families).
  Repo updated (dspark_proposer.py, patch_dspark_v1, patch_full_qall, FINDINGS §8, RESULTS DSpark table).
- 2026-09-12 00:40 PT — REAP-272E (LibertAIDAI/DeepSeek-V4.1-Flash-REAP-272E): 272 of 384 routed experts kept
  per layer (+4.1% text ppl per card), no Engram shards in the repo (base shards 47/48 linked in by
  link-reap-engram.sh), 207.6 GiB download in progress (dl-reap.log). Experts 4.76 GiB/layer -> a 10,10,10,10
  partition fits every expert on the GPUs (~52 GiB/rank). Cuts at 10 and 30 sit inside index groups; the PP
  shadow plan covers them by replaying the index source with top-k (rank 1 shadows 8; rank 3 shadows 20 + 28
  with top-k) — first real use of an in-group cut, so outputs get compared against an 8,12,8,12 boot.
  Launcher generalized (DSV41_HF_REPO), wrapper run-dsv41reap-pp4.sh (container dsv41reap-pp, :8005,
  DSv41ReapFlash, CPU layers none; 38-39 under dspark), servers-top/web entries DSv41R added.
- 2026-09-12 04:50 PT — REAP-272E UP on 10,10,10,10 with every expert on the GPUs (dsv41reap-pp :8005,
  DSv41ReapFlash). Two fixes: Engram shard links must be relative (the cache is /hf inside the container);
  vLLM's Triton DSv4 top-k router only admitted 256/384 experts and fell back to a CUDA kernel with a fixed
  expert table ("Unsupported expert number: 272") -> patch_reap_router.py admits any count (the Triton kernel
  pads to a power of two and masks). KV pool 4.68M tokens (was 904k). n-gram K=3 defaults: prose 28.3 tok/s
  incl. prefill (37 ms/step streaming), code-edit 43.3 tok/s incl. a 525-token prefill (~1 s; ~500 tok/s
  prefill), streaming code 129 steps in 6.0 s (~66 tok/s decode); code-edit greedy output identical to the
  base model's. lpcheck vs base eager: same texts, distributions flatter (up to 2.5-4.3 nats on 'Paris'/code
  probes) = the pruning. Validating the in-group cuts (index-source shadows 8 / 20+28) against 8,12,8,12 next.
- 2026-09-12 05:10 PT — 10,10,10,10 validated vs 8,12,8,12 (same probe texts, code-edit identical, prose
  diverges at char 429 = numerics); needle 30k RECALL OK with the index-source shadows, prefill 316 tok/s at
  29k. DSpark on REAP (CPU layers 38-39): prose 27.2 tok/s (n-gram 28.3), code-edit 48.2 (43.3), streaming
  code 8.4 s vs 6.0 s (2 CPU layers x ~20 ms per 6-row step), FRESH code 300 tokens in 5.7 s (53 tok/s,
  ~2x n-gram). Trying DSpark with a single CPU layer (39).
- 2026-09-12 05:30 PT — REAP FINAL: DSpark default on 10,10,11,9 with NO CPU experts (rank 3 = 9 layers + the
  8.4 GiB draft; cut 31 covered by the 28 index-source shadow). Fresh-boot verification: specbench prose 29-34
  tok/s, code-edit 51-94 tok/s incl. prefill; streaming prose 10.3-12.0 s / 400 tok, code-edit 4.3 s (72
  steps), fresh code 3.4 s / 300 tok; needle 30k RECALL OK; code-edit output identical to the base model.
  KV pool 1.42M tokens. dsv41reap-pp left running on :8005 (dsv41-pp stopped; only one can run).
- 2026-09-12 05:50 PT — PREFILL bottleneck: with a drafter on, the engine blocked in take_draft_token_ids after
  every batch, so prefill chunks went through the 4 ranks one at a time (1.4k tok/s flat from 6k to 21k tokens,
  ~1.47 s per 2048-token chunk = sum of the four ranks). core.py now skips the draft RPC for batches that do
  not complete a prompt (DSV41_DRAFT_SKIP_PREFILL=1): 2.4k tok/s at 5.9k, 3.1k tok/s at 14k tokens; decode
  and outputs unchanged (code-edit identical, 98% accepted). Remaining ceiling: per-rank chunk compute
  (~650 ms per 2048 tokens on the slowest rank) and the batch-queue depth of 4.
- 2026-09-12 07:20 PT — REAP FINAL (all layers on the GPUs, per the user): n-gram K=3, partition 10,10,10,10,
  CPU layers none, util 0.93 (rank 3 59.1 GiB used; 61-63 GiB has crashed here), max-model-len 524288 (KV pool
  2.22M tokens = 4.2 x 512k; 3.72M at 0.95). Needle 30k/200k RECALL OK, code-edit identical, prose streaming
  14.2 s / 400 tok, prefill 2.9k tok/s at 10k. DSpark needs 8.4 GiB on rank 3 -> only with 38-39 on the CPU
  (256k) or with an 11-layer rank at 131k; both rejected (CPU layers / margins). 1M max-model-len: the
  profiling transient plus an 11-layer rank or the draft leaves no KV memory. Standalone repo
  /home/r/dsv41reap-pp -> github.com/zebgop-ops/dsv41reap-pp. Soak 600 s x4 running.
