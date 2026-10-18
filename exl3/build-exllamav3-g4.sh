#!/bin/bash
# Variant build of exllamav3 for sm_80 with the fused MoE kernel capped at 4 expert groups (4 x 8 = 32 of the
# 70 SMs co-resident instead of 64 of 70), so the kernel's spinning group barriers cannot deadlock against
# kernels running on other streams (attention aux streams, NCCL). Output: exl3/wheel-g4, unpacked to exl3/site-g4.
set -eu
HERE=/home/r/dsv41-run/exl3; IMG=${DSV41_IMG:-vllm/vllm-openai:deepseekv41-flash-0909}
rm -rf "$HERE/src/exllamav3-g4" && cp -r "$HERE/src/exllamav3" "$HERE/src/exllamav3-g4"
H="$HERE/src/exllamav3-g4/exllamav3/exllamav3_ext/quant/exl3_devctx.cuh"
grep -q "define MOE_MAX_GROUPS 64" "$H" && sed -i 's/#define MOE_MAX_GROUPS 64/#define MOE_MAX_GROUPS 4   \/\/ dsv41: 4 x 8 SMs co-resident (was 64)/' "$H"
grep -n "MOE_MAX_GROUPS" "$H"
mkdir -p "$HERE/wheel-g4"
docker run --rm -e CUDA_VISIBLE_DEVICES= -e TORCH_CUDA_ARCH_LIST=8.0 -e MAX_JOBS=${MAX_JOBS:-12} -e CUDA_HOME=/usr/local/cuda \
  -v "$HERE/src/exllamav3-g4:/src:ro" -v "$HERE/wheel-g4:/wheel" --entrypoint bash "$IMG" -c '
    set -e; cp -r /src /build && cd /build
    PY_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
    export CPATH="$PY_SITE/nvidia/cu13/include${CPATH:+:$CPATH}" CPLUS_INCLUDE_PATH="$PY_SITE/nvidia/cu13/include" C_INCLUDE_PATH="$PY_SITE/nvidia/cu13/include"
    pip wheel --no-deps --no-build-isolation --no-cache-dir -w /wheel . 2>&1 | tail -3; ls -la /wheel'
rm -rf "$HERE/site-g4" && mkdir -p "$HERE/site-g4" && cd "$HERE/site-g4" && python3 -c "import zipfile,glob;zipfile.ZipFile(glob.glob('$HERE/wheel-g4/*.whl')[0]).extractall('.')" && ls
