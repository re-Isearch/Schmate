# Query energy at the vector-score boundary

For signed integer PASS cosine/IP spaces, energy is computed from the same
packed decoder as the inner product:

```
E(q) = sum(q[i] * q[i])
```

HNSW insertion and traversal continue using the existing fixed denominator
`C = 3932`. This preserves the stored graph and avoids shared query state:

```
native_distance = 1 - dot(q, x) / C
returned_distance = 1 - (C / E(q)) * (1 - native_distance)
                  = 1 - dot(q, x) / E(q)
```

The scale is local to one search. For positive energy this is a positive affine
transform, so it preserves neighbor ordering before floating-point rounding.
Neither HNSW's API nor the index/vector serialization format changes. The
calculated energy also uses the correct signed decoding for INT2/3/4/5/6/8/16,
including odd INT4 dimensions and components spanning byte boundaries. Binary,
floating-point, non-PASS quantizers, and L2 retain their native calibration.

A zero-energy query returns neutral distance `1`, rather than dividing by zero
or substituting an arbitrary positive energy. Its source score is `0.5`, which
IB's default calibration turns into zero semantic evidence. Energy normalization
does not repair a quantizer that has encoded the query as all zeroes.

Heap, ordered, filtered, and stop-condition searches use the conversion.
Stop-condition epsilon is now measured in the returned, calibrated distance
domain. That can change how many candidates a fixed epsilon admits, even though
top-K ordering is preserved. The stop-condition implementation still changes
the shared HNSW `ef` temporarily; this patch does not make that existing path
safe for concurrent calls. Ordinary top-K searches require no shared mutation.

PASS continues to bypass floating-point rescoring. When rescoring is used for
other quantizers, cosine/IP now returns `1 - similarity`, matching native
distances and their consumers, rather than `-similarity`. Both the UnifiedIndex
and BertIndex score conversions consistently return
`clamp((2 - distance) / 2, 0, 1)` for cosine/IP. This also fixes the reversed IP
score conversion. A record score therefore uses the same `(1 + similarity) / 2`
convention across these paths. These corrected source scores can affect existing
score filters; review tuned filters and epsilon settings on this branch.

`dot(q,x)/E(q)` is query-magnitude calibration, not full cosine:
candidate energy is not in the denominator, and similarity can exceed one.
Returned distances retain that magnitude; the public bounded score clamps it.
Cross-modality relevance still requires field/model-specific calibration and
validation. IB's Energy branch supplies stable score anchors and an optional
background-CDF curve for that next stage.

## Diagnostics

`utils/vector_background.cpp` consumes row-major native float32 embeddings for
one model and field, normalizes them as the cosine index does, and uses the same
INT4 PASS quantizer. It reports energy mean, deviation, coefficient of variation,
range and zero count to stderr. It samples distinct ordered pairs and writes
their bounded, query-energy-normalized source scores to stdout.

Build the diagnostic through the normal project, or independently:

```sh
c++ -std=c++20 -O2 -pthread -DNO_MANUAL_VECTORIZATION -include functional \
  -Iinclude utils/vector_background.cpp src/Logger.cpp -o vector_background
./vector_background 384 abstracts.f32 100000 42 > background-scores.txt
python3 ../ib/utils/vector_background.py --field ABSTRACT background-scores.txt
```

Paste the emitted settings into the database profile after choosing the noise
and strong score anchors. Sampling random pairs is an estimate of background
geometry, not a labeled relevance model. Use the exact same vector population,
quantization and score boosts as production; this diagnostic specifically covers
cosine INT4 PASS without indexer boosts.

## Verification

`bash tests/run_energy_tests.sh` builds the real UnifiedIndex source without an
embedding model dependency. It covers signed energy decoding, unchanged native
distance/order, INT4/IP/cosine and INT8/IP searches, heap/ordered/filter/epsilon
paths, zero-energy queries, concurrent ordinary searches, rescoring, and L2.
It uses portable kernels to avoid the repository's existing x86 SSE-dispatch
build issue. Small logger include/string-conversion corrections allow these
checks to compile with GCC.
