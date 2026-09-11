#!/bin/bash
# CPU-only import check of the overlaid modules inside the image (no GPU needed).
IMG=${DSV41_IMG:-vllm/vllm-openai:deepseekv41-flash-0909}
RUNDIR=${DSV41_RUNDIR:-$(cd "$(dirname "$0")/.." && pwd)}
MOUNTS=()
while IFS= read -r f; do rel=${f#"$RUNDIR/overlay/vllm/"}; MOUNTS+=(-v "$f:/usr/local/lib/python3.12/dist-packages/vllm/$rel:ro"); done < <(find "$RUNDIR/overlay/vllm" -type f -name '*.py')
docker run -i --rm "${MOUNTS[@]}" -v "$RUNDIR/overlay/engram_ssd:/opt/dsv41/engram_ssd:ro" -v "$RUNDIR/overlay/hybrid:/opt/dsv41/hybrid:ro" -v "$RUNDIR/kt/site:/opt/dsv41/kt-site:ro" \
  -e PYTHONPATH=/opt/dsv41:/opt/dsv41/kt-site -e CUDA_VISIBLE_DEVICES= -e VLLM_TARGET_DEVICE=cpu --entrypoint python3 "$IMG" - <<'PY'
import importlib, traceback
mods = [
  "vllm.v1.attention.ops.fp8_sm80", "vllm.v1.attention.ops.mqa_logits_triton",
  "vllm.model_executor.layers.sparse_attn_indexer", "vllm.v1.attention.ops.rocm_aiter_mla_sparse",
  "vllm.models.deepseek_v4_1.common.ops.cache_utils", "vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache",
  "vllm.models.deepseek_v4_1.common.ops.indexer_k_store", "vllm.models.deepseek_v4_1.common.engram",
  "vllm.models.deepseek_v4_1.amd.rocm", "vllm.models.deepseek_v4_1.ampere.ampere_sparse",
  "vllm.models.deepseek_v4_1.nvidia.model", "vllm.v1.attention.backends.registry",
  "engram_ssd.engram_ssd", "hybrid.cpu_experts", "hybrid.pp_shadow", "kt_kernel",
]
bad = 0
for m in mods:
    try:
        importlib.import_module(m); print("OK  ", m)
    except Exception as e:
        bad += 1; print("FAIL", m, "->", type(e).__name__, str(e)[:200])
print("failures:", bad)
PY
