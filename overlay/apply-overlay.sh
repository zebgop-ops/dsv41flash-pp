#!/bin/bash
# Apply every overlay patch script to overlay/vllm/ (seeded from the image by make-overlay.sh).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
V="$HERE/vllm"
python3 "$HERE/patch_engram.py"          "$V/models/deepseek_v4_1/common/engram.py"
python3 "$HERE/patch_select_attn.py"     "$V/models/deepseek_v4_1/nvidia/model.py"
python3 "$HERE/patch_registry.py"        "$V/v1/attention/backends/registry.py"
python3 "$HERE/patch_v41_ops_fp8.py"     "$V"
python3 "$HERE/patch_rocm_sparse_lut.py" "$V/v1/attention/ops/rocm_aiter_mla_sparse.py"
python3 "$HERE/patch_indexer.py"         "$V/model_executor/layers/sparse_attn_indexer.py"
python3 "$HERE/patch_attention_v41.py"   "$V/models/deepseek_v4_1/attention.py"
python3 "$HERE/patch_fused_indexer_q.py"  "$V/models/deepseek_v4/common/ops/fused_indexer_q.py"
python3 "$HERE/patch_cpu_experts.py"     "$V"
python3 "$HERE/patch_pp_shadow.py"       "$V/models/deepseek_v4_1/nvidia/model.py"
python3 "$HERE/patch_debug_stats.py"     "$V/models/deepseek_v4_1/nvidia/model.py"
python3 "$HERE/patch_mhc_torch.py"       "$V/models/deepseek_v4_1/nvidia/model.py"
python3 "$HERE/patch_misc_sm80.py"       "$V"
python3 "$HERE/patch_weight_iter.py"     "$V/model_executor/model_loader/weight_utils.py"
python3 "$HERE/patch_kv_groups.py"       "$V/v1/core/kv_cache_utils.py"
python3 "$HERE/patch_mla_indexer.py"     "$V/v1/attention/backends/mla/indexer.py"
python3 "$HERE/patch_v1_circular.py"     "$V/v1/worker/gpu_model_runner.py"
python3 "$HERE/patch_marlin_staged.py"   "$V/model_executor/layers/quantization/mxfp4.py"
for f in $(find "$V" -name '*.py'); do python3 -c "import ast,sys;ast.parse(open('$f').read())" || { echo "SYNTAX ERROR: $f"; exit 1; }; done
echo "overlay applied: $(find "$V" -name '*.py' | wc -l) files"
