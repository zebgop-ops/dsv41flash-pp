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

## Speculative decoding (n-gram prompt lookup)

`DSV41_SPEC=K` turns on vLLM's n-gram drafter (`prompt_lookup_min` 5, `max` 8 by default; no draft
model, rejection sampling). Same launcher, same checkpoint; all numbers single-stream, greedy,
400 generated tokens, wall time including prefill (`tools/specbench.py`, thinking on) or streaming
with thinking off (`tools/itl.py`).

| workload | no speculation | K=3 | K=2 |
|---|---|---|---|
| code edit, 525-token prompt (thinking on): drafts accepted | – | 97% (2.91 tokens/step) | 97% (1.94 tokens/step) |
| code edit: wall incl. prefill | 31.6-32.2 s (12.4-12.7 tok/s) | 26.2 s (15.3 tok/s) | 27.9 s (14.3 tok/s) |
| code edit (thinking off), 529-token prompt, streaming | – | 23.3 s (129 steps for 400 tokens) | 27.9 s (14.3 tok/s)ITL |
| free prose (thinking on): drafts accepted | – | 80% of 15 drafted (5 draft steps in 400 tokens) | 92% of 12 drafted (6 draft steps) |
| free prose: wall incl. prefill | 23.8-24.5 s (16.3-16.8 tok/s) | 28.3 s (14.1 tok/s) | 27.9 s (14.3 tok/s) |
| free prose (thinking off), streaming, two runs | 21.8 / 25.2 s | 26.1 / 27.9 s | 27.9 s (14.3 tok/s)ITL |
| draft-less decode step (streaming windows) | 58-60 ms | 61-70 ms | 70-72 ms |
| top-5 logprobs vs eager, 3 probes × 3 positions | 0.000 | 0.000 | 0.000 |
| greedy code-edit output vs no-spec | – | identical | identical |

Where the time goes: a 1-token step costs ~65 ms with speculation on versus ~58 ms off, because
vLLM disables async scheduling for CPU n-gram drafting (FINDINGS.md §7); a 1+K-token verify step
costs ~2.5x a 1-token step because the CPU-expert layers stream every expert any row routes to.
Speculation therefore wins where drafts are accepted most of the time (code, edits, quoting) and
loses ~10% on free prose.

### DSpark (the checkpoint's own drafter, `DSV41_SPEC_METHOD=dspark`)

Streaming, thinking off, 300 generated tokens, wall time including prefill. DSpark runs with
layers 33-39 on the CPU (two more than the other rows) and confidence truncation at 0.7.

| workload | no speculation | n-gram K=3 | DSpark |
|---|---|---|---|
| free prose | ~18 s | ~20 s | 22.2 s (1.44 tokens/step, 106 ms/step) |
| code edit, 529-token prompt (copy-heavy) | ~31 s | ~23 s | 20.6 s (5.56 tokens/step) |
| fresh code (LRU cache + tests, nothing to copy) | ~19 s | ~19 s | 15.3 s (5.1 tokens/step, 85% of drafts accepted) |
| logprobs vs eager | 0.000 | 0.000 | text same; 0.5-0.9 nats at later positions (multi-row verify numerics) |
| greedy code-edit output vs no-spec | – | identical | identical |

Without truncation (`DSV41_DSPARK_CONF=0`) every step verifies 6 rows at ~265 ms: prose falls to
8.3 tok/s. The cost is the CPU-expert layers (~20 ms each per 6-row step, FINDINGS.md §8).

### REAP-272E, every expert on the GPUs (`serve/run-dsv41reap-pp4.sh`, :8005)

Same prompts as above, streaming with thinking off unless noted. The default is DSpark on a
10,10,11,9 partition (the draft sits on the 9-layer rank); n-gram uses 10,10,10,10. No CPU experts
either way. Numbers from a fresh boot after one warm-up request per prompt shape.

| workload | base model, n-gram (9 CPU layers) | REAP, n-gram K=3 | REAP, DSpark (default) |
|---|---|---|---|
| free prose, 400 tokens streaming | 26-28 s (14 tok/s) | 12.7-14.2 s (28-31 tok/s) | 10.3-12.0 s (33-39 tok/s) |
| code edit, 529-token prompt, 400 tokens streaming | 23.5 s | 6.0 s | 4.3 s (72 steps, ~110 tok/s decode) |
| fresh code (LRU cache + tests), 300 tokens streaming | 15.3 s (DSpark) | ~11 s | 3.4 s (63 steps, 4.7 tokens/step) |
| specbench prose / code-edit (thinking on, incl. prefill) | 13.6 / 14.8 tok/s | 28.3 / 43.3 tok/s | 29-34 / 51-94 tok/s |
| prefill, 525-token prompt | ~60 tok/s | ~500 tok/s | ~500 tok/s |
| prefill, 6k / 14k-token prompts (chunks pipelined, FINDINGS §9) | ~100 tok/s | – | 2.4k / 3.1k tok/s |
| prefill, 29k-token prompt (needle, recall OK; before the pipelining fix) | ~106 tok/s | 316 tok/s | 316 tok/s |
| KV pool at 131k max-model-len | 904k tokens | 4.7M tokens | 1.4M tokens |
| greedy code-edit output vs the base model | – | identical | identical |

Pruning cost per the model card: +4.1% text perplexity. On the three logprob probes the pruned
model produces the same texts with flatter distributions (up to 2.5-4.3 nats on the top token).
The 10,10,10,10 layout was checked against 8,12,8,12 (identical code-edit output, prose diverging
at char 429, i.e. the stack's usual kernel numerics) and with the 30k needle test.

### DeepSeek-V4.1-Flash-1017, abliterated (`serve/run-dsv41-1017-pp4.sh`, :8007)

`mrfakename/DeepSeek-V4.1-Flash-1017-abliterated`: DeepSeek's 2026-10-17 weights refresh (same config,
tokenizer and shard layout as 0909; all 48 shards differ) with a rank-2 refusal subspace projected out
of `attn.wo_b` and `ffn.shared_experts.w2`; shards 1-2 and 43-48 (embeddings, DSpark layers, Engram
tables) are the official 1017 bytes (Hub SHA-256 per shard). Served at utilization 0.93 (FINDINGS §4,
§10): 12 cold layers' experts on the CPU (7, 15-19, 27, 35-39) so that each rank keeps 7 GPU expert
layers, prefill chunk 1088 tokens, context 262,144. Same prompts as above, streaming with thinking
off unless noted, n-gram K=3.

| workload | 0909 base, 9 CPU layers, util 0.97 | 1017, 12 CPU layers, util 0.93 |
|---|---|---|
| free prose, 400 tokens streaming | 26-28 s (14 tok/s) | 30.6 s (13.1 tok/s) |
| code edit, 529-token prompt, 400 tokens streaming | 23.5 s | 27.2 s (123 steps, 97% drafts accepted) |
| specbench prose / code-edit (thinking on, incl. prefill) | 13.6 / 14.8 tok/s | 12.7 / 13.2 tok/s |
| KV pool | 904k tokens at 131k max-model-len | 4.26M tokens at 262k (16x a 256k request) |
| GPU used per rank while serving | 58.8 / 58.6 / 57.6 / 53.5 GiB weights, up to ~61 GiB | 56.3 / 57.7 / 57.1 / 60.1 GiB |
| after profiling, before KV | – | 54.3 / 53.8 / 53.2 / 56.8 GiB |
| host RAM | 81 GB serving, 87 GB peak | 99 GB serving, 107 GB peak |

The ~15% decode cost is the three extra CPU layers (~2.4 ms each per 1-row step, more per verify
row); the 10-layer attempt (7,16-19,35-39) measured 53.9 / 60.3 / 59.4 / 56.2 GiB after profiling and
vLLM found no KV room on ranks 1-2 at 0.93.

Quality so far (same harness and inputs as the 0909 base, `tools/quality.py`): wikitext-2 perplexity
2.753 vs 2.715 for the 0909 base (+1.4%, 67,131 tokens, 36 windows; the weights refresh and the
abliteration edit are not separable here). Code perplexity, MMLU-1000 and HumanEval+/MBPP+ were
interrupted when the GPUs were needed elsewhere and are pending (`tools/eval-1017.sh`,
`tools/eval-1017-code.sh` resume them against a running server).

### DeepSeek-V4.1-Flash-1017 EXL3 2.9 bpw, every expert on the GPUs (`serve/run-dsv41-exl3-pp4.sh`, :8008)

`LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` through exllamav3 1.6.3 kernels built for sm_80
(FINDINGS §11): routed experts K=3 (layers 18-22 K=2) in the fused trellis MoE kernel, attention /
shared experts / compressor / indexer / head as EXL3 linears, Engram tables from the MXFP4 1017
snapshot. Partition 10,10,10,10, no CPU experts, utilization 0.91, 262,144-token window, n-gram K=3,
`FULL_AND_PIECEWISE` graphs. Same prompts as above, streaming with thinking off unless noted.

| workload | MXFP4 1017, 12 CPU layers | EXL3 1017, all-GPU | REAP-272E, n-gram (reference) |
|---|---|---|---|
| free prose, 400 tokens streaming | 30.6 s (13.1 tok/s) | 24.0 s (16.7 tok/s, 64 ms/step) | 12.7-14.2 s (28-31 tok/s) |
| code edit, 529-token prompt, 400 tokens streaming | 27.2 s | 12.0 s (123 steps, 97% drafts accepted) | 6.0 s |
| specbench prose / code-edit (thinking on, incl. prefill) | 12.7 / 13.2 tok/s | 12.5 / 25.6 tok/s | 28.3 / 43.3 tok/s |
| prefill 1.5k / 6k / 12k-token prompts | ~180-200 tok/s | 0.5k / 1.1k / 0.8k tok/s (measured beside an MMLU run) | 2.4k / 3.1k tok/s |
| KV pool | 4.26M tokens at 262k | 4.39M tokens at 262k | 4.7M at 131k |
| GPU used per rank while serving | 56.3 / 57.7 / 57.1 / 60.1 GiB (util 0.93) | 58.2 / 53.5 / 52.3 / 58.4 GiB (util 0.91) | up to 59 GiB (util 0.93) |
| host RAM | 99 GB | ~16 GB | ~16 GB |
| load to first request | ~7 min (CPU expert repack) | ~2.5 min (weights 70 s, graphs 3 s) | ~3 min |
| greedy code-edit output vs the MXFP4 1017 | – | identical (prose diverges at char 124) | identical to base |

Quality, same harness and inputs as the other rows of this file (`tools/quality.py`, paired per window):

| metric | 0909 base | MXFP4 1017 | EXL3 1017 | REAP-272E (0909) |
|---|---|---|---|---|
| wikitext-2 perplexity (67,131 tokens, 36 windows) | 2.7147 | 2.7530 (+1.4% vs base) | 2.8503 (+3.5% vs MXFP4 1017, per-window nll +0.035 ± 0.007) | 3.4821 (+28.3%) |
| Python stdlib code perplexity (30,589 tokens, 16 windows) | 1.3633 | not measured | 1.3775 (+1.0% vs base, +0.010 ± 0.006) | 1.3642 |
| MMLU-1000 (same questions, cloze) | 84.4% | not measured (interrupted) | 83.6% (−0.8 pts; McNemar p = 0.37, 927/1000 same answers, margin 3.40 → 2.92 nats) | 76.7% (p < 1e-4) |
| HumanEval+ / MBPP+ pass@1 | 93.9% / 84.9% | – | running (`tools/eval-1017-code.sh exl3`) | 93.9% / 83.3% |

The pack's card reports +0.4% prose / +1.8% code perplexity on its own held-out sets; on wikitext the
3-bit trellis costs +3.5% here, an order of magnitude less than REAP's expert pruning (+28%), and the
MMLU difference to the unquantized 0909 base is not significant over 1000 paired questions (REAP's −7.7
points was, at p < 1e-4).

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

- Speculation is n-gram prompt lookup (on by default, `DSV41_SPEC=0` to turn off). DSpark, the
  model's own drafter, stays off: this image has no `broadcast_drafts` for PP, the draft stack would
  need ~6.5 GiB on rank 3, and its verify steps would pay the same CPU-expert cost per row.
- Async scheduling is off whenever speculation is on (vLLM disables it for CPU n-gram), which is
  the ~7 ms/step cost on draft-less steps (FINDINGS.md §7).
- Context validated to 29k tokens; the launcher allows 131k and the model 1M. Prefill time
  (~65 tok/s) is the practical limit, not memory.
- The chat template has no thinking toggle; reasoning always comes back in
  `message.reasoning`, so budget `max_tokens` for it.
- 16 threads for the CPU experts (one per physical core); 32 (SMT) and active OpenMP waiting
  were both slower. The kernel is within ~15 % of the host's DRAM read bandwidth, so the
  remaining lever for decode is fewer CPU layers, not a faster kernel.
