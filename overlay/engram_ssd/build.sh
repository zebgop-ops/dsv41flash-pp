#!/bin/bash
# Build librow_store.so next to row_store.cpp. Run inside the serving image (same glibc as
# the workers) or on any host whose glibc is not newer than the image's.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
g++ -O2 -Wall -Wextra -std=c++17 -shared -fPIC -pthread "$HERE/row_store.cpp" -o "$HERE/librow_store.so"
echo "built $HERE/librow_store.so"
