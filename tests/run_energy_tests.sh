#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$repo/build"
test_build=$(mktemp -d "$repo/build/energy-tests.XXXXXX")
trap 'rm -rf "$test_build"' EXIT
cxx=${CXX:-c++}
# Compile the headers normally so missing includes or stale APIs fail here,
# instead of being hidden by forced standard-header includes.
flags=(-std=c++20 -O1 -g -pipe -pthread
       -ffunction-sections -fdata-sections -I"$repo/include")
linker_flag=-Wl,--gc-sections
if [[ $(uname -s) == Darwin ]]; then linker_flag=-Wl,-dead_strip; fi
objects=()
sanitize_flags=()
if [[ ${SANITIZE:-} == address ]]; then
    sanitize_flags=(-fsanitize=address -fno-omit-frame-pointer)
fi
for source_file in src/unified_hnsw.cpp src/HnswConfig.cpp src/Logger.cpp src/FileLock.cpp; do
    object_file="$test_build/${source_file##*/}.o"
    TMPDIR="$test_build" "$cxx" "${flags[@]}" "${sanitize_flags[@]}" -c "$repo/$source_file" -o "$object_file"
    objects+=("$object_file")
done
for layout in canonical legacy; do
    layout_flags=()
    if [[ $layout == legacy ]]; then
        layout_flags=(-DSCHMATE_TEST_LEGACY_HEADERS_FIRST)
    fi
    TMPDIR="$test_build" "$cxx" "${flags[@]}" "${sanitize_flags[@]}" "${layout_flags[@]}" \
        "$repo/tests/header_layout.cpp" "${objects[@]}" "$linker_flag" -o "$test_build/header_layout_$layout"
    (cd "$test_build" && "./header_layout_$layout")
done
for test_name in vector_energy vector_persistence; do
    TMPDIR="$test_build" "$cxx" "${flags[@]}" "${sanitize_flags[@]}" -c "$repo/tests/$test_name.cpp" -o "$test_build/$test_name.o"
    "$cxx" -pthread "${sanitize_flags[@]}" "$test_build/$test_name.o" "${objects[@]}" "$linker_flag" -o "$test_build/$test_name"
    (cd "$test_build" && "./$test_name")
done
