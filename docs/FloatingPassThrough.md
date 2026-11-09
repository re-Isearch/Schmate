# Floating-point PASS distances

FP16 and BF16 are floating-point storage formats, not signed 16-bit integer
codes. The old PASS inner-product path selected a decoder from bit width alone:
it treated FP16/BF16 exponent and mantissa bits as integer magnitudes and divided
their dot product by the signed-code reference energy. Unrelated vectors could
produce huge negative distances, which the public score mapping clamped to 1.
The L2 PASS path also selected four-bit nibbles for these floating formats.

FP16, BF16, and FP32 PASS now decode their actual values before computing:

```
IP distance = 1 - sum(a[i] * b[i])
squared L2 distance = sum((a[i] - b[i])^2)
```

Cosine searches retain UnifiedIndex's existing normalization of input floats.
Floating PASS does not apply signed-integer query-energy calibration or the
integer reference denominator. Signed Q4/Q8 calibration is unchanged. Packed
unaligned addresses and FP16 subnormal decoding are covered by the tests.
The stored floating encodings and index header format do not change. New graph
rows are padded to keep subsequent link lists aligned when the packed vector
has an odd byte count or an odd number of 16-bit components. The header already
stores the row stride and preserves this padding on reload.

## Existing indexes

Rebuild FP16/BF16 PASS graphs constructed with the broken distance function
after updating: their stored values are intact, but their graph links were
built using the wrong metric. Rebuild explicit quantized-space FP32 PASS graphs
if they used that same path; ordinary unquantized FP32 indexes used float
kernels already and are unaffected.

Older graphs with unaligned packed row strides need rebuilding to receive the
new row padding; loading does not rewrite their stored rows.

## Regression

Run `bash tests/run_energy_tests.sh`, or build the focused model-independent
Q8 and floating PASS tests:

```sh
cmake -S tests/q8 -B build-pass
cmake --build build-pass
ctest --test-dir build-pass --output-on-failure
```

The floating test covers exact representable values, signs, FP16 subnormals,
unaligned buffers, odd dimensions, graph row alignment before/after reload,
IP/L2 units, and the production
ordered/heap/filtered/epsilon search and save/reload paths at 768D and 1024D.
It checks unrelated vectors against float-reference distances so clamped
all-one scores cannot hide a broken metric. No model weights are required.
