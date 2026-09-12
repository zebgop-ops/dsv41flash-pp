#!/bin/bash
# DeepSeek-V4.1-Flash (native FP8 dense / MXFP4 experts / FP8 Engram) on 4x CMP 170HX (sm_80,
# 64 GiB, PCIe Gen2 x4, no P2P): PIPELINE PARALLEL 4 + cold-layer experts on the CPU (kt-kernel
# AVX2 MXFP4) + Engram tables served from the NVMe checkpoint shards. See PLAN.md.
#
# IMAGE: official vllm/vllm-openai:deepseekv41-flash-0909 plus a Python-only overlay bind-mounted
#   from overlay/vllm/ (pristine copies: overlay/image-official/).
# Knobs: DSV41_PARTITION DSV41_CPU_EXPERT_LAYERS DSV41_ENGRAM_STORAGE(ssd|host) DSV41_MAXLEN DSV41_SEQS
#        DSV41_UTIL DSV41_CG DSV41_SPEC DSV41_PORT DSV41_NAME DSV41_IMG DSV41_EXTRA_ARGS DSV41_GPUS
#        DSV41_EXTRA_DOCKER (extra docker-run flags, e.g. "-e CUDA_LAUNCH_BLOCKING=1")
set -u
NAME=${DSV41_NAME:-dsv41-pp}
PORT=${DSV41_PORT:-8004}
SERVED=${DSV41_SERVED:-DSv41Flash}
IMG=${DSV41_IMG:-vllm/vllm-openai:deepseekv41-flash-0909}
HFCACHE=${DSV41_HF:-/home/r/.cache/huggingface}
RUNDIR=/home/r/dsv41-run
HF_REPO=${DSV41_HF_REPO:-deepseek-ai/DeepSeek-V4.1-Flash}   # e.g. LibertAIDAI/DeepSeek-V4.1-Flash-REAP-272E (see run-dsv41reap-pp4.sh)
REPO_DIR="$HFCACHE/hub/models--${HF_REPO//\//--}"
SNAPSHOT=${DSV41_SNAPSHOT:-$(ls "$REPO_DIR/snapshots" 2>/dev/null | head -1)}
MODEL="/hf/hub/models--${HF_REPO//\//--}/snapshots/$SNAPSHOT"
# Layer partition (see PLAN.md "PP4 layout"): cuts must sit on kv-source (2,8,14,20) or
# index-source (24,28,32,36) boundaries; cuts at 24+ get a shadow of kv source 20.
PARTITION=${DSV41_PARTITION:-8,12,8,12}
PP=$(echo "$PARTITION" | tr "," "\n" | wc -l)
# Layers whose 384 routed experts run on the CPU (kt-kernel). Decoder layers only (they do
# not run on prompt tokens once SWA-bounded replay lands; today they do, so prefill pays).
# ngram (prompt lookup, no draft model; the default) | dspark (the checkpoint's own 3-layer parallel drafter,
# ported to the V1 runner: +20% on fresh code, -15% on prose here, see FINDINGS.md §8). DSpark keeps a
# confidence-truncated draft (DSV41_DSPARK_CONF, cumulative acceptance confidence; /dump/dspark_conf overrides
# it at runtime, 0 = always verify all 5).
SPEC_METHOD=${DSV41_SPEC_METHOD:-ngram}
if [ "$SPEC_METHOD" = dspark ]; then
  CPU_LAYERS=${DSV41_CPU_EXPERT_LAYERS:-16-19,33-39}   # two more CPU layers on rank 3: the draft (7.4 GiB + 1 GiB embedding) lives there
  SPEC_N=${DSV41_SPEC:-5}                              # = dspark_block_size
else
  CPU_LAYERS=${DSV41_CPU_EXPERT_LAYERS:-16-19,35-39}   # 8 GPU expert layers per rank fit since the Marlin repack is staged through host RAM
  SPEC_N=${DSV41_SPEC:-3}              # n-gram speculative tokens per step (0 = off; 3 measured best on code, ~10% slower on free prose)
fi
ENGRAM=${DSV41_ENGRAM_STORAGE:-ssd}
GPU_ORDER=${DSV41_GPUS:-all}
UTIL=${DSV41_UTIL:-0.97}
MAXLEN=${DSV41_MAXLEN:-131072}
SEQS=${DSV41_SEQS:-4}
# Graph mode: without speculation FULL_DECODE_ONLY (exact, no piecewise graphs needed). With speculation
# FULL_AND_PIECEWISE: 1-token steps replay q=1 FULL graphs (patch_full_q1), verify steps q=K+1 FULL graphs,
# mixed batches piecewise (Engram/shadow sources need patch_engram_piecewise there); prefill runs eager.
if [ "$SPEC_N" != "0" ]; then CG=${DSV41_CG:-FULL_AND_PIECEWISE}; else CG=${DSV41_CG:-FULL_DECODE_ONLY}; fi
EXTRA_ARGS=${DSV41_EXTRA_ARGS:-}
PATCHDIR=${DSV41_PATCH:-$RUNDIR/overlay/vllm}
MEM_CAP=${DSV41_MEM_CAP_FRACTION:-0.965}   # clean OOM instead of an Xid-31 wedge near the top of a card

# ---------------- preflight ----------------
if [ -z "$SNAPSHOT" ] || [ ! -f "$REPO_DIR/snapshots/$SNAPSHOT/model-00048-of-00048.safetensors" ]; then
  echo "checkpoint not complete in $REPO_DIR (snapshot='$SNAPSHOT'; the REAP repos need the base model's Engram shards 47/48 linked in)"; exit 1
fi
if ls "$REPO_DIR/blobs/"*.incomplete >/dev/null 2>&1; then
  echo "download still in progress (incomplete blobs present)"; exit 1
fi
KMOD=$(sed -n 's/.*Kernel Module *\([0-9.]*\).*/\1/p' /proc/driver/nvidia/version | head -1)
ULIB=$(dpkg-query -W -f='${Version}' libnvidia-compute-610 2>/dev/null | cut -d- -f1)
if [ -n "$KMOD" ] && [ -n "$ULIB" ] && [ "$KMOD" != "$ULIB" ]; then
  echo "NVIDIA driver mismatch: kernel module $KMOD vs userland $ULIB -> new CUDA containers cannot start; reboot first"; exit 1
fi
for other in dsv4-a100 qwen38-pp qwen38u-pp glm53flash-pp glm53nvfp4-pp dsv41-pp dsv41reap-pp; do
  if [ "$other" != "$NAME" ] && docker ps --format '{{.Names}}' | grep -qx "$other"; then
    echo "refusing to start: $other is running on the same GPUs (stop it yourself first)"; exit 1
  fi
done
if ! docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --entrypoint python3 "$IMG" \
     -c 'import torch;[torch.randn(8,8,device=f"cuda:{i}") for i in range(torch.cuda.device_count())]' >/dev/null 2>&1; then
  echo "CUDA init failed on at least one GPU (wedged driver?) -> reboot before starting"; exit 1
fi

docker stop -t 60 "$NAME" >/dev/null 2>&1; docker rm "$NAME" >/dev/null 2>&1

# ---------------- overlay mounts ----------------
MOUNTS=()
if [ -d "$PATCHDIR" ]; then
  while IFS= read -r f; do
    rel=${f#"$PATCHDIR"/}
    MOUNTS+=(-v "$f:/usr/local/lib/python3.12/dist-packages/vllm/$rel:ro")
  done < <(find "$PATCHDIR" -type f -name '*.py')
fi
MOUNTS+=(-v "$RUNDIR/overlay/engram_ssd:/opt/dsv41/engram_ssd:ro" -v "$RUNDIR/overlay/hybrid:/opt/dsv41/hybrid:ro" -v "$RUNDIR/kt/site:/opt/dsv41/kt-site:ro")

# Tensors the GPU loader must not even mmap: Engram tables in SSD mode, and the experts of
# the CPU layers (kt-kernel reads those itself).
SKIP_RE=""
if [ "$ENGRAM" = ssd ]; then SKIP_RE='layers\.(1|14)\.engram\.embed\.'; fi
CPU_RE=$(python3 - "$CPU_LAYERS" <<'PY'
import sys
out=set()
for part in sys.argv[1].split(','):
    part=part.strip()
    if not part or part.lower() == 'none': continue   # DSV41_CPU_EXPERT_LAYERS=none -> every expert on the GPUs
    if '-' in part:
        a,b=part.split('-'); out.update(range(int(a),int(b)+1))
    else: out.add(int(part))
print('|'.join(str(x) for x in sorted(out)))
PY
)
if [ -n "$CPU_RE" ]; then SKIP_RE="${SKIP_RE:+$SKIP_RE|}layers\.($CPU_RE)\.ffn\.experts\."; fi
SPEC_ARGS=()
# SPEC_METHOD is set with CPU_LAYERS above (dspark needs two more CPU layers on the last rank)
if [ "$SPEC_N" != "0" ]; then
  # capture sizes 1..4 (q=1 graphs, one per request count) plus multiples of K+1 (verify steps); vLLM's own
  # rounding to multiples of K+1 is disabled by patch_full_q1 (DSV41_FULL_Q1=1)
  if [ -z "${DSV41_CG_SIZES:-}" ]; then
    # q=1..K families for 1-2 requests, K+1 family for all request counts (patch_full_qall)
    DSV41_CG_SIZES=$(python3 -c "k=$SPEC_N+1; print(sorted(set(range(1,$SEQS+1))|{q*r for q in range(2,k) for r in (1,2)}|{m*k for m in range(1,$SEQS+1)}))" | tr -d ' ')
  fi
  SPEC_ARGS+=("-cc.cudagraph_capture_sizes=$DSV41_CG_SIZES")
  if [ "$SPEC_METHOD" = ngram ]; then
    # prompt_lookup_min 5: fewer failed drafts on free prose (a 1+K-row verify step costs ~2.5x a 1-row step
    # because the CPU experts stream every routed expert); code/edit workloads accept ~97% either way
    SPEC_ARGS+=(--speculative-config "{\"method\":\"ngram\",\"num_speculative_tokens\":$SPEC_N,\"prompt_lookup_max\":${DSV41_NGRAM_MAX:-8},\"prompt_lookup_min\":${DSV41_NGRAM_MIN:-5}}")
  else
    SPEC_ARGS+=(--speculative-config "{\"method\":\"$SPEC_METHOD\",\"num_speculative_tokens\":$SPEC_N}")
  fi
fi

echo "starting $NAME: partition $PARTITION, CPU expert layers [$CPU_LAYERS], engram=$ENGRAM, maxlen $MAXLEN"
docker run -d --name "$NAME" --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$GPU_ORDER" \
  --ipc=host --shm-size 32g -p "$PORT:$PORT" \
  -v "$HFCACHE:/hf" "${MOUNTS[@]}" \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/opt/dsv41:/opt/dsv41/kt-site \
  -e VLLM_PP_LAYER_PARTITION="$PARTITION" \
  -e DSV41_CPU_EXPERT_LAYERS="$CPU_LAYERS" -e DSV41_ENGRAM_STORAGE="$ENGRAM" \
  -e DSV41_ENGRAM_ZERO="${DSV41_ENGRAM_ZERO:-0}" -e DSV41_DEBUG_STATS="${DSV41_DEBUG_STATS:-0}" -e DSV41_MHC_TORCH="${DSV41_MHC_TORCH:-0}" -e DSV41_DEBUG_DUMP="${DSV41_DEBUG_DUMP:-}" -e DSV41_CPU_MOE="${DSV41_CPU_MOE:-native}" -e DSV41_DEBUG_TIMING="${DSV41_DEBUG_TIMING:-0}" -e DSV41_DEBUG_PROFILE="${DSV41_DEBUG_PROFILE:-0}" -e DSV41_DEBUG_GDUMP="${DSV41_DEBUG_GDUMP:-}" -e DSV41_DEBUG_SPEC="${DSV41_DEBUG_SPEC:-0}" -e DSV41_DEBUG_ENGRAM="${DSV41_DEBUG_ENGRAM:-0}" -e DSV41_PREFILL_EAGER="${DSV41_PREFILL_EAGER:-1}" -e DSV41_FULL_Q1="${DSV41_FULL_Q1:-1}" -e DSV41_DEBUG_CORE="${DSV41_DEBUG_CORE:-0}" -e DSV41_DEBUG_TRACE="${DSV41_DEBUG_TRACE:-0}" -e DSV41_DEBUG_DSPARK="${DSV41_DEBUG_DSPARK:-0}" -e DSV41_DSPARK_CONF="${DSV41_DSPARK_CONF:-0.7}" -e DSV41_DEBUG_SYNC="${DSV41_DEBUG_SYNC:-0}" -e DSV41_CPU_EXPERT_THREADS="${DSV41_CPU_EXPERT_THREADS:-16}" -v /home/r/dsv41-run/dump:/dump -e DSV41_MODEL_DIR="$MODEL" -e DSV41_MEM_CAP_FRACTION="$MEM_CAP" -e DSV41_SKIP_WEIGHT_RE="$SKIP_RE" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_USE_V2_MODEL_RUNNER=0 -e PYTHONFAULTHANDLER=1 \
  ${DSV41_EXTRA_DOCKER:-} \
  "$IMG" \
  "$MODEL" --served-model-name "$SERVED" --port "$PORT" \
  --pipeline-parallel-size "$PP" --gpu-memory-utilization "$UTIL" \
  --max-model-len "$MAXLEN" --max-num-seqs "$SEQS" --max-num-batched-tokens 2048 \
  --attention-backend TRITON_MLA_SPARSE_DSV41 --kv-cache-dtype fp8 --attention-config '{"indexer_kv_dtype":"fp8"}' \
  -cc.cudagraph_mode="$CG" --no-enable-prefix-caching \
  "${SPEC_ARGS[@]}" \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice \
  $EXTRA_ARGS
echo "docker logs -f $NAME  (ready on 'Application startup complete.')"
