# DeepSeek-V4.1-Flash on 4× sm_80: native quants, CPU-offloaded experts, Engram on NVMe

A runnable serving setup, a Python-only vLLM overlay, and the forensics behind it for
**[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)**
(552B backbone + 196B Engram n-gram tables, 40 layers, CSA2 sparse attention, mHC
hyper-connections) on hardware and at a memory budget it was never meant for:
4× **CMP 170HX** (GA100 silicon, sm_80, 64 GB each after the VRAM unlock, PCIe Gen2 x4, no
NVLink/P2P), a 16-core Zen3 host with 123 GB RAM, and one NVMe.

The checkpoint is served in its **native formats** (block-FP8 dense weights, MXFP4 routed
experts, FP8 Engram rows): no requantization. What does not fit on the cards is split three
ways:

- **GPU:** 31 layers' experts (8 per rank) + all attention, Marlin MXFP8/MXFP4 kernels,
  pipeline parallel 4 (partition 8,12,8,12).
- **CPU RAM:** the routed experts of 9 layers (16-19, 35-39) run on the host through a small
  native AVX2/OpenMP MXFP4 MoE kernel (`overlay/hybrid/cpu_moe.cpp`) that reads the HF shards
  directly (~65 GB of RAM); kt-kernel remains selectable (`DSV41_CPU_MOE=kt`).
- **NVMe:** the two 94.6 GiB Engram tables are never loaded. Rows are `pread` from the
  original safetensors shards through the page cache inside a CUDA host callback, so the
  lookup sits in the CUDA graph like any other op.

Headline numbers (details in **[RESULTS.md](RESULTS.md)**): correct answers with reasoning,
exact planted-value recall at 29k tokens, **16.7 tok/s single-stream decode, ~100 tok/s
prefill**, 933k-token KV pool at 131k context. It is a 750B model on four mining cards, and
it is correct.

The official `vllm/vllm-openai:deepseekv41-flash-0909` image (vLLM PR
[#56214](https://github.com/vllm-project/vllm/pull/56214) lineage) only has Hopper/Blackwell
and ROCm attention paths. Everything here is a bind-mount overlay on that image: no rebuild,
no CUDA code, one 200-line C++ file for the row store.

Sister projects (same box, same conventions):
[glm53flash-pp](https://github.com/zebgop-ops/glm53flash-pp),
[qwen38-flashnext-pp](https://github.com/zebgop-ops/qwen38-flashnext-pp).

## Quickstart

```bash
docker pull vllm/vllm-openai:deepseekv41-flash-0909
hf download deepseek-ai/DeepSeek-V4.1-Flash                 # 475 GiB, 48 shards, into the HF cache
pip download kt-kernel==0.7.0.post2 gguf typer rich -d kt/wheel   # then unpack the wheels into kt/site
docker run --rm -v $PWD/overlay/engram_ssd:/w --entrypoint bash vllm/vllm-openai:deepseekv41-flash-0909 /w/build.sh
docker run --rm -v $PWD/overlay/hybrid:/w --entrypoint bash vllm/vllm-openai:deepseekv41-flash-0909 /w/build.sh
./overlay/import-check.sh                                    # CPU-only import test of every overlaid module
DSV41_HF=$HOME/.cache/huggingface ./serve/run-dsv41-pp4.sh   # serves DSv41Flash on :8004
```

`kt/site` must contain the unpacked `kt_kernel` package plus its `gguf`, `typer` and `rich`
dependencies (`pip install --target kt/site kt_kernel-0.7.0.post2-*.whl gguf typer rich`, then
delete any `deep_gemm` it drags in). The launcher mounts it read-only into the image.

`serve/run-dsv41-pp4.sh` is the exact production launcher, with every flag, env var and
bind-mount. Knobs: `DSV41_PARTITION` (default `8,12,8,12`), `DSV41_CPU_EXPERT_LAYERS`
(`16-19,35-39`), `DSV41_CPU_MOE` (`native` | `kt`), `DSV41_CPU_EXPERT_THREADS` (16, one per
physical core), `DSV41_ENGRAM_STORAGE` (`ssd`), `DSV41_MAXLEN` (131072), `DSV41_SEQS`,
`DSV41_UTIL` (0.97), `DSV41_SPEC` (n-gram speculative tokens per step, default `3`; `0` = off;
`DSV41_NGRAM_MIN` 5 / `DSV41_NGRAM_MAX` 8; `DSV41_SPEC_METHOD=dspark` for the checkpoint's own
drafter, FINDINGS.md §8), `DSV41_CG` (`FULL_DECODE_ONLY` without speculation,
`FULL_AND_PIECEWISE` with it; `NONE` for diagnostics), `DSV41_GPUS`, `DSV41_EXTRA_ARGS`,
`DSV41_EXTRA_DOCKER` (e.g. `"-e CUDA_LAUNCH_BLOCKING=1"`), and the diagnostic switches in
[FINDINGS.md](FINDINGS.md). It pre-flights the checkpoint, the
driver (kernel module vs userland mismatch after an upgrade), other servers on the cards,
and CUDA init on every GPU before touching anything.

## What had to change, and why

Details and the debugging story are in **[FINDINGS.md](FINDINGS.md)**; the working log is
[PLAN.md](PLAN.md). The overlay is `overlay/vllm/` (mounted file by file over the image's
`vllm` package); `patches/vllm-overlay.diff` is the full delta against the pristine image
for review; `overlay/patch_*.py` + `overlay/apply-overlay.sh` are the anchor-based scripts
that produced it.

1. **Ampere attention path.** `DeepseekV41AmpereMLAAttention` subclasses the PR's ROCm
   Triton sparse-MLA attention (ragged sparse prefill/decode, bf16 o_proj einsum) and runs it
   on CUDA; every fp8 conversion inside the Triton kernels goes through LUT/RNE helpers
   because Triton refuses `fp8e4nv` below SM89 (same trick as the V4 port in
   haosdent/vllm@f8ea5bb). DeepGEMM is absent, so the DSA indexer gets a Triton MQA-logits
   fallback, a torch top-k fallback, and row-chunked logits.
2. **Engram on NVMe** (`overlay/engram_ssd/`). A new storage mode for
   `ParallelEngramEmbedding`: hash ids → D2H → `cudaLaunchHostFunc` → thread-pool `pread` of
   FP8 rows + UE8M0 scales straight from shards 47/48 → pinned staging → H2D → dequant on the
   GPU. Design adapted from 0xSero's row store (MIT); rewritten with a thread pool over the
   page cache.
3. **CPU experts** (`overlay/hybrid/cpu_experts.py`, `cpu_moe.cpp`). For the configured layers
   the routed experts are created on the meta device and skipped by the loader; the MoE
   runner's forward runs the gate, router and shared expert on the GPU and hands top-k ids
   and weights to the host through pinned staging and a `cudaLaunchHostFunc` node, so the
   call sits inside the CUDA graph. The native kernel (AVX2/FMA + OpenMP, E2M1 nibbles
   decoded through a byte LUT, fp32 accumulate, exact against a torch reference) streams the
   FP4 weights at ~44 GB/s of this host's ~52 GB/s: 2.4 ms per layer-step at batch 1.
   kt-kernel's AVX2 MXFP4 path, which it replaces, runs one thread per selected expert and
   takes 5.7 ms regardless of thread count (and needs its pinned buffers registered for
   graph capture sizes).
4. **Pipeline parallel across kv-sharing groups** (`overlay/hybrid/pp_shadow.py`). V4.1
   consumers read their kv-source layer's compressed KV and indexer K caches on the same
   rank, and the decoder group (layers 20-39) does not fit one card. A receiving rank gets a
   *shadow* of the source layer's attention (compressor + indexer, 0.13 GiB) registered under
   the source's prefixes, driven by the source's post-norm attention input shipped in
   `IntermediateTensors`, plus layer 20's candidate blocks. Index-source shadows replay the
   top-k too, so any index-source boundary is a legal cut.
5. **Two bugs that produced garbage or crashes** (FINDINGS.md): the MXFP8 *emulation* linear
   kernel (the only bmm-capable one below SM90) leaves its E8M0 scale on `wo_a` after
   dequantizing it, and the ROCm einsum helper applied it a second time; and the V1 model
   runner (required for Engram lookback) ran the generic token→slot kernel on the
   compressor's circular-buffer cache group, reading past its one-block-per-request table
   (an MMU fault on the CMP at ~2k prompt tokens, silent garbage reads on Hopper).
6. **Marlin repack without the load-time transient** (`overlay/hybrid/marlin_staged.py`).
   vLLM's MXFP4 MoE prepare keeps raw and packed expert tensors on the GPU at once (~6.7 GiB
   per layer, the raw ones pinned by the caller frames), which capped a 64 GB card at 7
   expert layers. Staging the raw tensor through host RAM and packing expert by expert (bit-
   exact against vLLM's prepare) lets 8 layers per rank load, so only 9 layers' experts stay
   on the CPU. The staging must be pageable: torch's pinned-host allocator caches freed
   blocks for the process lifetime and the four workers OOM'd the host.
7. **Smaller upstream fixes:** KV-cache tensor builder iterating other ranks' layers of a
   projected group under PP; `has_deep_gemm()` vs `is_deep_gemm_supported()` gating in the
   MLA indexer metadata builder; tilelang prenorm and `execute_in_parallel` capture guards;
   a Triton autotune guard for keys that surface only under full-graph capture.
8. **Speculative decoding under PP with CPU experts** (FINDINGS.md §7). vLLM's n-gram
   drafter (no draft model, rejection sampling) needed six fixes to run on this stack:
   non-last ranks had no `drafter` attribute; the scheduler shipped only the one scheduled
   token to ranks 0-2 (accepted drafts never arrived); the PP batch queue scheduled a request
   again while its drafts were in flight; PIECEWISE graphs were captured with
   `attn_metadata=None`, baking Engram hashing and the shadow-source inserts in as no-ops;
   graph padding rows ran full CPU-expert passes; and vLLM rounds every capture size to a
   multiple of K+1 and knows one uniform-decode query length, so draft-less steps ran in
   piecewise graphs (`patch_full_q1.py` adds q=1 FULL graphs). Logprobs equal eager; the
   code-edit output is identical with and without speculation. The remaining cost of a
   draft-less step (~65 vs ~58 ms) is vLLM disabling async scheduling for CPU n-gram.
9. **DSpark, the checkpoint's own drafter, on the V1 runner** (FINDINGS.md §8). The image has
   it only for the V2 runner (no Engram there); `overlay/hybrid/dspark_proposer.py` ports the
   V2 speculator onto V1's DFlash machinery with per-KV-group metadata, PP-aware loading and
   confidence-truncated drafts. It works and accepts 85% of drafts on fresh code (+20%), but a
   6-row verify step streams ~40 experts per CPU layer from host RAM, so prose loses; n-gram
   stays the default.

## Layout

```
serve/run-dsv41-pp4.sh   production launcher (preflight, mounts, all flags)
overlay/vllm/            the Python overlay, mounted over the image's vllm package
overlay/patch_*.py       anchor-based, idempotent patch scripts; apply-overlay.sh runs them all
                         (spec decode under PP: patch_v1_spec_pp, patch_pp_spec_tokens, patch_engram_piecewise,
                         patch_prefill_eager, patch_pad_mask, patch_full_q1 — FINDINGS.md §7)
overlay/make-overlay.sh  extracts pristine files from the image (for diffing / re-seeding)
overlay/engram_ssd/      row_store.cpp + build.sh, engram_ssd.py, CPU/GPU tests
overlay/hybrid/          cpu_experts.py, cpu_moe.cpp/.py (native CPU MoE), marlin_staged.py,
                         pp_shadow.py (PP shadow sources), build.sh, tests
overlay/sm80-src/        the V4 Ampere files (haosdent/vllm@f8ea5bb) the port was regexed from
patches/vllm-overlay.diff  full delta vs the pristine image
ktests/                  kernel tests and the torch oracles (ref_layer0*.py, ref_stages.py)
kt/                      kt-kernel MXFP4 correctness/perf tests (wheel and site dir not included)
tools/                   smoke.py, plen.py, needle.py, bench.py, soak.py (+ detect.py);
                         specbench.py (spec-decode acceptance + output diff), lpcheck.py (top-5 logprobs vs
                         a saved eager run), itl.py (streaming per-window rate), batchinv.py (batch
                         invariance of the eager path), tracetl.py / profcpu.sh (timeline + profiler readers)
PLAN.md                  design notes and the boot-by-boot log
```

Tests run inside the image: `docker run --rm --gpus '"device=0"' -v $HOME/.cache/huggingface:/hf:ro
-v $PWD/ktests:/ktests:ro <mounts of overlay/vllm> --entrypoint python3 vllm/vllm-openai:deepseekv41-flash-0909
-m pytest /ktests/...` (the oracles need the checkpoint at `/hf` and the stage dumps from a
`DSV41_DEBUG_DUMP=/dump` run).

## Credits

vLLM (Apache-2.0) and the DeepSeek V4.1 PR authors for the model definitions and the ROCm
Triton path; haosdent for the V4 Ampere port this follows; 0xSero for the NVMe row-store
design; kvcache-ai for kt-kernel; DeepSeek for the reference `inference/model.py` the oracles
re-implement. Everything in this repo is Apache-2.0 (see LICENSE); `row_store.cpp` carries
its MIT attribution.
