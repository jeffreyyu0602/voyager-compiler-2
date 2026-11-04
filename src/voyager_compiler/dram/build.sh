#!/bin/bash
# Build the DRAMSim3 shim the reporting estimator loads under --dram_sim:
# third_party/DRAMsim3 plus shim.cc, into build/libvoyager_dramsim3.so here.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
DRAMSIM3="$ROOT/third_party/DRAMsim3"
OUT="$HERE/build"
mkdir -p "$OUT"

"${CXX:-g++}" -std=c++11 -O3 -shared -fPIC -DFMT_HEADER_ONLY=1 \
    -I"$DRAMSIM3/src" -I"$DRAMSIM3/ext/headers" -I"$DRAMSIM3/ext/fmt/include" \
    "$DRAMSIM3"/src/*.cc "$HERE/shim.cc" \
    -o "$OUT/libvoyager_dramsim3.so"
echo "Built $OUT/libvoyager_dramsim3.so"
