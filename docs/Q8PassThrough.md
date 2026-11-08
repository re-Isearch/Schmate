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

It stores one byte per component, preserving both signs. Q8 inner-product
distance reverses this scale; Q8 cosine distance uses the norms of the stored
signed vectors; Q8 squared L2 reads all eight bits and returns the result in
the original vector units. Q4, FP16, and FP32 encodings retain their formats.

## Existing Q8 indexes

Rebuild indexes created with the broken Q8 packer from their original input.
Updating the executable cannot recover components already stored as zero or
reconstruct graph edges built from those collapsed vectors.

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
IP/cosine/L2 distances, float-reference comparisons, nearest-neighbor ranking,
and graph save/reload at 768 and 1024 dimensions. Q4, FP16, and FP32 encoding
checks guard their existing byte formats. The tested Q8 path is shared by
the scalar and Apple Silicon builds; it does not use the dormant unsigned
INT8 SIMD helper functions.
