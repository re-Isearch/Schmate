#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$repo/build"
test_build=$(mktemp -d "$repo/build/energy-tests.XXXXXX")
trap 'rm -rf "$test_build"' EXIT
cxx=${CXX:-c++}
# Use the portable kernels. Explicit standard-header includes accommodate the
# existing headers' transitive include assumptions on GCC as well as Clang.
flags=(-std=c++20 -O1 -g -pipe -pthread -DNO_MANUAL_VECTORIZATION
       -include atomic -include cstring -include functional
       -ffunction-sections -fdata-sections -I"$repo/include")
linker_flag=-Wl,--gc-sections
if [[ $(uname -s) == Darwin ]]; then linker_flag=-Wl,-dead_strip; fi
objects=()
for source_file in tests/vector_energy.cpp src/unified_hnsw.cpp src/Logger.cpp src/FileLock.cpp; do
    object_file="$test_build/${source_file##*/}.o"
    TMPDIR="$test_build" "$cxx" "${flags[@]}" -c "$repo/$source_file" -o "$object_file"
    objects+=("$object_file")
done
"$cxx" -pthread "${objects[@]}" "$linker_flag" -o "$test_build/vector_energy"
(cd "$test_build" && ./vector_energy)
