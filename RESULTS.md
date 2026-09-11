# Results

All numbers from the production launcher defaults (`serve/run-dsv41-pp4.sh`: PP=4, partition
`7,7,10,16`, routed experts of layers 21-23 and 31-39 on the CPU, Engram tables served from the
NVMe checkpoint shards, `PIECEWISE` CUDA graphs, `--max-model-len 131072`, `--max-num-seqs 4`,
`--max-num-batched-tokens 2048`), measured 2026-09-10/11 on 4x CMP 170HX (GA100, sm_80, 64 GB
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
| single-stream decode (chat, ~40-token prompt) | 8.1 tok/s with CUDA graphs (~5 tok/s eager) |
| 4 concurrent streams, 300 tokens each (`tools/bench.py 4`) | 15.8 tok/s aggregate, 3.9 tok/s per stream |
| prefill, 5.5k-token prompt | ~62 tok/s |
| prefill, 29k-token prompt | ~67 tok/s (436 s to first token) |

Where the time goes: kt-kernel's AVX2 MXFP4 MoE costs ~5.4 ms per layer-step at batch 1
(19-26 GB/s of weight traffic from RAM), so the 12 CPU-expert layers alone put the decode
ceiling around 15 tok/s; the four PCIe Gen2 x4 hops and the Triton (no DeepGEMM / FlashMLA)
attention path take the rest. Prefill runs the CPU experts on every prompt token of those 12
layers (2048-token chunks), which is what keeps it near 65 tok/s.

## Memory

| rank | layers | GPU weights + non-torch | KV pool |
|---|---|---|---|
| 0 | 0-6 (+ embed, Engram layer 1) | 51.8 GiB | shared pool: 2,298,226 tokens total, 17.5x a 131k request |
| 1 | 7-13 (shadow of index source 2) | 50.4 GiB | |
| 2 | 14-23 (experts of 21-23 on CPU, Engram layer 14) | 51.6 GiB | |
| 3 | 24-39 (experts of 31-39 on CPU, shadow of kv source 20, lm_head) | 54.0 GiB | |

Host RAM: ~8.3 GB per CPU-expert layer (packed FP4 + fp32-converted scales) x 12 = ~100 GB,
plus ~1.5 GB per worker; 113 GB of 123 GB in use while serving. The two 94.6 GiB Engram
tables are never loaded: rows are `pread` from shards 47/48 through the page cache
(exact rows, ~4 M rows/s warm in the CPU test).

## Not done / known limits

- DSpark speculative decoding stays off: this image has no `broadcast_drafts` for PP.
- Context validated to 29k tokens; the launcher allows 131k and the model 1M. Prefill time
  (~65 tok/s) is the practical limit, not memory.
- The chat template has no thinking toggle; reasoning always comes back in
  `message.reasoning`, so budget `max_tokens` for it.
- 16 threads for the CPU experts (one per physical core) was faster than 32 in the
  standalone kt-kernel test; no further tuning yet.
