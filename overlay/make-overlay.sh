#!/bin/bash
# Extract pristine copies of the vLLM files the overlay touches from the image into
# overlay/image-official/ (for diffing) and seed overlay/vllm/ with copies to edit.
set -euo pipefail
IMG=${DSV41_IMG:-vllm/vllm-openai:deepseekv41-flash-0909}
HERE=$(cd "$(dirname "$0")" && pwd)
SITE=/usr/local/lib/python3.12/dist-packages/vllm
FILES=(
  models/deepseek_v4_1/common/engram.py
  models/deepseek_v4_1/common/ops/cache_utils.py
  models/deepseek_v4_1/common/ops/fused_compress_quant_cache.py
  models/deepseek_v4_1/common/ops/indexer_k_store.py
  models/deepseek_v4_1/common/ops/query_quant.py
  models/deepseek_v4_1/common/ops/__init__.py
  models/deepseek_v4_1/amd/rocm.py
  models/deepseek_v4_1/amd/model.py
  models/deepseek_v4_1/nvidia/model.py
  models/deepseek_v4_1/nvidia/vl_model.py
  models/deepseek_v4_1/nvidia/dspark.py
  models/deepseek_v4_1/nvidia/model_state.py
  models/deepseek_v4_1/attention.py
  models/deepseek_v4_1/compressor.py
  models/deepseek_v4_1/sparse_mla.py
  models/deepseek_v4_1/quant_config.py
  models/deepseek_v4/nvidia/model.py
  models/deepseek_v4/common/ops/fused_indexer_q.py
  models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py
  model_executor/layers/sparse_attn_indexer.py
  model_executor/kernels/mhc/tilelang.py
  model_executor/layers/quantization/mxfp4.py
  model_executor/layers/quantization/modelopt.py
  model_executor/kernels/linear/mxfp8/marlin.py
  model_executor/kernels/linear/mxfp8/__init__.py
  v1/attention/backends/mla/sparse_swa.py
  v1/attention/backends/mla/indexer.py
  v1/attention/backends/registry.py
  v1/attention/ops/rocm_aiter_mla_sparse.py
  v1/worker/gpu/model_runner.py
  v1/worker/gpu/pp_utils.py
  v1/worker/gpu_worker.py
  v1/worker/gpu_model_runner.py
  v1/cudagraph_dispatcher.py
  v1/engine/core.py
  v1/core/sched/scheduler.py
  config/engram.py
  config/vllm.py
  config/speculative.py
  config/attention.py
  engine/arg_utils.py
  platforms/cuda.py
  utils/import_utils.py
  utils/multi_stream_utils.py
  model_executor/model_loader/weight_utils.py
  v1/worker/utils.py
  v1/core/kv_cache_utils.py
  v1/attention/backends/mla/indexer.py
  utils/deep_gemm.py
)
CID=$(docker create "$IMG")
trap 'docker rm "$CID" >/dev/null' EXIT
for f in "${FILES[@]}"; do
  mkdir -p "$HERE/image-official/$(dirname "$f")"
  if docker cp "$CID:$SITE/$f" "$HERE/image-official/$f" 2>/dev/null; then
    if [ ! -f "$HERE/vllm/$f" ]; then mkdir -p "$HERE/vllm/$(dirname "$f")"; cp "$HERE/image-official/$f" "$HERE/vllm/$f"; fi
  else
    echo "missing in image: $f"
  fi
done
docker cp "$CID:$SITE/_version.py" "$HERE/image-official/_version.py" 2>/dev/null || true
echo "done: $(find "$HERE/image-official" -name '*.py' | wc -l) files"
