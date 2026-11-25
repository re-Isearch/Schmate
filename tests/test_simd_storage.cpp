// Cross-architecture, byte-exact IntStorage regression.
// Compile normally on x86 with -mavx2, on AArch64 with NEON, and without SIMD.
// Expected bytes are produced independently of the implementation under test.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <vector>
#include "hnswlib/int_storage.h"

static void check(bool ok, const char* why, size_t dim = 0) {
    if (!ok) {
        std::cerr << "int_storage SIMD parity failed: " << why << ", dimension " << dim << '\n';
        std::exit(1);
    }
}

static uint8_t expected_int4(float value) {
    // Signed nibble, low component first. Round-to-nearest with halfway
    // values away from zero after clamping, independent of CPU rounding mode.
    const float scaled = std::clamp(value * 64.0f, -8.0f, 7.0f);
    return static_cast<uint8_t>(static_cast<int>(std::round(scaled)) & 15);
}

static void verify_int4(size_t dim) {
    std::vector<float> values(dim);
    const float fixtures[] = {
        0.0f, 0.5f/64, -0.5f/64, 1.5f/64, -1.5f/64,
        7.0f/64, -8.0f/64, 7.5f/64, -8.5f/64,
        1.0f, -1.0f, 100.0f, -100.0f,
        0.25f/64, -0.25f/64, 4.5f/64, -4.5f/64
    };
    for (size_t i = 0; i < dim; ++i) values[i] = fixtures[(i * 7 + dim) % (sizeof(fixtures)/sizeof(fixtures[0]))];

    const size_t bytes = (dim + 1) / 2;
    std::vector<uint8_t> actual(bytes + 3, 0xa5), expected(bytes, 0);
    for (size_t i = 0; i < dim; ++i) {
        const uint8_t nibble = expected_int4(values[i]);
        if (i & 1) expected[i / 2] |= uint8_t(nibble << 4);
        else expected[i / 2] |= nibble;
    }
    hnswlib::IntStorage::quantize(hnswlib::StorageType::INT4, values.data(), actual.data(), dim);
    check(std::equal(expected.begin(), expected.end(), actual.begin()), "INT4 float differs from scalar reference", dim);
    check(actual[bytes] == 0xa5 && actual[bytes + 1] == 0xa5, "INT4 wrote past packed vector", dim);

    // The double-input overload must preserve the same format rather than
    // using float-specific SIMD loads against a double*.
    std::vector<double> doubles(values.begin(), values.end());
    std::fill(actual.begin(), actual.end(), 0xa5);
    hnswlib::IntStorage::quantize(hnswlib::StorageType::INT4, doubles.data(), actual.data(), dim);
    check(std::equal(expected.begin(), expected.end(), actual.begin()), "INT4 double differs from float reference", dim);
    check(actual[bytes] == 0xa5, "INT4 double wrote past packed vector", dim);
}

static void verify_fp16() {
    // A numerical cast of float16_t to uint16_t (as opposed to a bitwise
    // store) incorrectly turns half(0.5) into integer zero on ARM.
    const float vals[] = {0.5f, -0.5f, 1.0f, -1.0f, 2.0f, 0.0f};
    const uint16_t expected[] = {0x3800, 0xb800, 0x3c00, 0xbc00, 0x4000, 0x0000};
    uint8_t packed[sizeof(expected) + 4];
    std::memset(packed, 0xa5, sizeof(packed));
    hnswlib::IntStorage::quantize(hnswlib::StorageType::FP16, vals, packed, 6);
    for (size_t i = 0; i < 6; ++i) {
        uint16_t bits = 0;
        std::memcpy(&bits, packed + i * 2, 2);
        check(bits == expected[i], "FP16 encoded numerically instead of storing half bits", i);
    }
    check(packed[sizeof(expected)] == 0xa5, "FP16 wrote beyond output");
}

int main() {
    for (size_t dim : {size_t(1), size_t(2), size_t(3), size_t(7),
                       size_t(8), size_t(9), size_t(15), size_t(16),
                       size_t(17), size_t(31), size_t(32), size_t(33),
                       size_t(63), size_t(64), size_t(65), size_t(768),
                       size_t(769), size_t(1025)})
        verify_int4(dim);
    verify_fp16();
    std::cout << "IntStorage SIMD/scalar parity passed\n";
}
