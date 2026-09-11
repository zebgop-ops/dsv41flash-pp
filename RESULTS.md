# Results

All numbers from the production launcher defaults (`serve/run-dsv41-pp4.sh`: PP=4, partition
`8,12,8,12`, routed experts of layers 16-19 and 35-39 on the CPU via the native kernel, Engram
tables served from the NVMe checkpoint shards, `FULL_AND_PIECEWISE` CUDA graphs,
`--gpu-memory-utilization 0.97`, `--max-model-len 131072`, `--max-num-seqs 4`,
`--max-num-batched-tokens 2048`), measured 2026-09-11 on 4x CMP 170HX (GA100, sm_80, 64 GB
each after the VRAM unlock, PCIe Gen2 x4, no P2P), a 16-core Threadripper PRO 5955WX (Zen3,
AVX2 only) with 123 GB RAM, and a Samsung 990 PRO NVMe holding the checkpoint.

## Correctness

| check | tool | result |
|---|---|---|
| "The capital of France is" (raw completion) | curl | ` Paris.` |
| 17 * 23, capital of Australia (chat, reasoning parser) | `tools/smoke.py` | `391`, `Canberra` |
| ~2k-token raw prompt, "last item number" | `tools/plen.py 1450` | correct (`240`) |
| planted code at 55 % depth, 5,567 tokens | `tools/needle.py 6000` | recalled |
| planted code at 55 % depth, 29,067 tokens | `tools/needle.py 30000` | recalled |

Per-stage oracle on layer 0 (vLLM stage dumps vs a torch re-implementation of DeepSeek's
reference `model.py`, `ktests/ref_stages.py`), cosine similarity after the wo_a fix:

| stage | cosine |
|---|---|
| q / kv projections + RMSNorm | 1.0000 |
| fused Q RoPE + KV RoPE/UE8M0 cache insert | 1.0000 |
| gathered fp8 K rows (dequantized) | 0.9998 |
| Triton sparse prefill kernel (given vLLM's own q and K) | 1.0000 |
| output projection (inverse RoPE + wo_a einsum + wo_b) | 1.0000 (was 0.92 with ~2^-11 magnitude) |
| layer-1 attention end to end | 0.9999 |
| layer-0/1 MoE (router + Marlin MXFP4 experts + shared expert) | 0.99999 |

## Speed

| workload | result |
|---|---|
| single-stream decode, steady state (engine 10 s windows, 600-800-token generations) | 16.6-16.9 tok/s |
| 600-token completion, wall time including prefill | 36.4 s (16.5 tok/s) |
| 4 concurrent streams, 300 tokens each (`tools/bench.py 4`) | 24.3 tok/s aggregate (25-26 steady), 6.1 tok/s per stream |
| prefill, 5.5k-token prompt | ~99 tok/s |
| prefill, 29k-token prompt | ~106 tok/s (275 s to first token) |

History of the single-stream number, same checkpoint and outputs throughout:

| configuration | decode |
|---|---|
| 12 CPU layers on kt-kernel, `PIECEWISE` graphs | 8.1 tok/s |
| same, `FULL_AND_PIECEWISE` graphs | 8.1 tok/s (graphs were not the bottleneck) |
| 12 CPU layers on the native kernel | 10.9 tok/s |
| 9 CPU layers (staged Marlin repack, 8 GPU expert layers per rank) | 16.7 tok/s |

Where a ~60 ms decode step goes (in-worker torch profiler + per-rank timing, `FINDINGS.md`):
~22 ms in the 9 CPU-expert layers (2.4 ms each; the kernel streams FP4 at ~44 GB/s against a
measured ~52 GB/s DRAM ceiling on this host), ~27 ms of GPU device time across the four
ranks (~0.6 ms per layer: Marlin dense/MoE GEMMs, the Triton sparse decode, tilelang mHC),
and the rest in the three PCIe hops, host-callback bubbles and scheduling. kt-kernel's AVX2
MXFP4 path, used before, cost 5.7 ms per layer-step at any thread count. Prefill runs the CPU
experts on every prompt token of the 9 layers (1.8 s per 2048-token chunk), which is what
holds it near 100 tok/s.

## Memory

| rank | layers | GPU weights + non-torch | KV pool |
|---|---|---|---|
| 0 | 0-7 (+ embed, Engram layer 1) | 58.8 GiB | shared pool: 933,214 tokens, 7.1x a 131k request |
| 1 | 8-19 (experts of 16-19 on CPU, Engram layer 14) | 58.6 GiB | |
| 2 | 20-27 (candidate source 20) | 57.6 GiB | |
| 3 | 28-39 (experts of 35-39 on CPU, shadow of kv source 20, lm_head) | 53.5 GiB | |

Host RAM: ~7.2 GB per CPU-expert layer (packed FP4 + E8M0 scales, the checkpoint's own
bytes) x 9 = ~65 GB, plus ~2 GB per worker; peak 87 GB during load (the staged repack
passes each expert tensor through pageable host memory), ~81 GB while serving. The two 94.6 GiB Engram
tables are never loaded: rows are `pread` from shards 47/48 through the page cache
(exact rows, ~4 M rows/s warm in the CPU test).

## Not done / known limits

- DSpark speculative decoding stays off: this image has no `broadcast_drafts` for PP.
- Context validated to 29k tokens; the launcher allows 131k and the model 1M. Prefill time
  (~65 tok/s) is the practical limit, not memory.
- The chat template has no thinking toggle; reasoning always comes back in
  `message.reasoning`, so budget `max_tokens` for it.
- 16 threads for the CPU experts (one per physical core); 32 (SMT) and active OpenMP waiting
  were both slower. The kernel is within ~15 % of the host's DRAM read bandwidth, so the
  remaining lever for decode is fewer CPU layers, not a faster kernel.
