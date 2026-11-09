# Q8 pass-through storage

GGUF Q8 model weights and the vectors stored in an HNSW index are separate
representations. The embedder still supplies float embeddings. Schmate's
INT8 pass-through storage packs those embedding components into signed bytes.

The broken packer rounded the original normalized components directly and
clamped them to unsigned 0..255. A typical component such as `0.03125` became
zero, and negative components were also discarded. Entire 768D/1024D vectors
could collapse to zeros. The resulting cosine distance was always one, which
the search display converted to a score of `0.500000` for every candidate.

The corrected encoding is symmetric signed fixed point:

```
q = clamp(round(component * 127), -127, 127)
```

It stores one byte per component, preserving both signs. Q8 squared L2 reads
all eight bits and returns the result in the original vector units. Q4, FP16,
and FP32 encodings retain their formats.

## Query-energy calibration

This branch integrates the Q8 storage fix with the October `Energy` work.
Signed PASS cosine/IP retains the fixed native HNSW distance and applies
query-energy calibration once, after traversal:

```
native_distance = 1 - dot(q, x) / reference_energy
returned_distance = 1 - dot(q, x) / E(q)
E(q) = dot(q, q)
```

The public source score is `clamp((2 - returned_distance) / 2, 0, 1)`.
Identical and opposite coded vectors produce scores 1 and 0; a zero-energy
query produces the neutral score 0.5. Candidate norms are not included in the
denominator. The earlier full-cosine Q8 distance from the main-branch hotfix is
superseded here, so query energy is not applied on top of a cosine-normalized
native metric. See [QueryEnergy.md](QueryEnergy.md) for the scoring contract.

`utils/vector_background.cpp` and its build target remain available with their
original INT4 PASS background-sampling behavior.

## Existing Q8 indexes

Rebuild indexes created with the broken Q8 packer from their original input.
Updating the executable cannot recover components already stored as zero or
reconstruct graph edges built from those collapsed vectors.

Also rebuild Q8 graphs created with the earlier full-cosine main-branch hotfix
when moving to this branch: their traversal metric differs even though their
signed Q8 byte encoding is the same.

## Model-independent regression

The focused test builds without bert.cpp, model weights, or an inference backend:

```sh
cmake -S tests/q8 -B build-q8
cmake --build build-q8
ctest --test-dir build-q8 --output-on-failure
```

It can also be enabled in the normal Schmate build with
`-DSCHMATE_BUILD_Q8_TESTS=ON`.

The test covers signed low-magnitude components, saturation, odd dimensions,
the fixed native IP metric, query-energy calibration, L2 units, float-reference
comparisons, and the production UnifiedIndex ordered/heap/epsilon searches and
save/reload at 768 and 1024 dimensions. A candidate with twice the query's coded
magnitude verifies that query-energy calibration is preserved rather than
replaced by full cosine. Q4, FP16, and FP32 checks guard their byte formats.

`bash tests/run_energy_tests.sh` also runs the Q8 regression alongside the
October energy, persistence, and header-layout checks. These tests compile
normally with native headers and SIMD dispatch; no forced standard headers or
disabled SIMD are required. The INT8 packer itself uses the shared scalar path,
including on Apple Silicon; the dormant unsigned INT8 SIMD helpers are unused.
