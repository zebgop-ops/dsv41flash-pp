#!/bin/bash
# Build exllamav3 (its CUDA extension for sm_80) inside the serving image, CPU only, into exl3/wheel/.
# usage: build-exllamav3.sh [git ref]   (default v1.6.3)
set -eu
REF=${1:-v1.6.3}; HERE=/home/r/dsv41-run/exl3; IMG=${DSV41_IMG:-vllm/vllm-openai:deepseekv41-flash-0909}
[ -d "$HERE/src/exllamav3/.git" ] || git clone -q https://github.com/turboderp-org/exllamav3.git "$HERE/src/exllamav3"
git -C "$HERE/src/exllamav3" fetch -q --tags origin && git -C "$HERE/src/exllamav3" checkout -q "$REF"
echo "exllamav3 at $(git -C "$HERE/src/exllamav3" rev-parse --short HEAD) ($REF)"
docker run --rm -e CUDA_VISIBLE_DEVICES= -e TORCH_CUDA_ARCH_LIST=8.0 -e MAX_JOBS=${MAX_JOBS:-14} -e CUDA_HOME=/usr/local/cuda \
  -v "$HERE/src/exllamav3:/src:ro" -v "$HERE/wheel:/wheel" --entrypoint bash "$IMG" -c '
    set -e; nvcc --version | tail -1; cp -r /src /build && cd /build
    PY_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
    export CPATH="$PY_SITE/nvidia/cu13/include${CPATH:+:$CPATH}" CPLUS_INCLUDE_PATH="$PY_SITE/nvidia/cu13/include" C_INCLUDE_PATH="$PY_SITE/nvidia/cu13/include"
    pip wheel --no-deps --no-build-isolation --no-cache-dir -w /wheel . 2>&1 | tail -5
    ls -la /wheel'
