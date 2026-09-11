#!/bin/bash
# Build libcpu_moe.so (AVX2/FMA/F16C + OpenMP). Run inside the serving image or on a host
# whose glibc is not newer than the image's.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
g++ -O3 -std=c++17 -shared -fPIC -fopenmp -mavx2 -mfma -mf16c -Wall -Wextra "$HERE/cpu_moe.cpp" -o "$HERE/libcpu_moe.so"
echo "built $HERE/libcpu_moe.so"
