// Both supported header paths must expose one storage/index API. Include the
// legacy path first as well: a stale class definition otherwise hides missing
// members until production sources such as unified_hnsw.cpp are compiled.
#ifdef SCHMATE_TEST_LEGACY_HEADERS_FIRST
#include "../unified/LSMVectorStorage.h"
#include "../unified/unified_hnsw.hpp"
#include "unified_hnsw.hpp"
#else
#include "unified_hnsw.hpp"
#include "../unified/LSMVectorStorage.h"
#include "../unified/unified_hnsw.hpp"
#endif

#include <cstdlib>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

int main()
{
    hnswlib::VectorStorageConfig config;
    config.mode = hnswlib::VectorStorageMode::IN_MEMORY;
    hnswlib::LSMVectorStorage storage(2, config);
    const float values[]{3.0f, 4.0f};
    storage.addPoint(7, values);
    float squared_norm = 0.0f;
    if (!storage.with_vector(7, [&](const float* vector) {
            squared_norm = vector[0] * vector[0] + vector[1] * vector[1];
        }) || squared_norm != 25.0f) {
        std::cerr << "Both header layouts must expose lock-protected vector access\n";
        return EXIT_FAILURE;
    }
    if (storage.with_vector(8, [](const float*) { std::abort(); })) {
        std::cerr << "A missing vector must not invoke the callback\n";
        return EXIT_FAILURE;
    }
    // Exercise small dimensions and SIMD blocks/residuals. The native header
    // must provide every function selected by InnerProductSpace.
    for (size_t dim : {size_t(1), size_t(2), size_t(3), size_t(4),
                       size_t(12), size_t(16), size_t(19), size_t(65)}) {
        std::vector<float> a(dim), b(dim);
        for (size_t i = 0; i < dim; ++i) {
            a[i] = float(int(i % 7) - 3) / 4.0f;
            b[i] = float(int(i % 5) - 2) / 8.0f;
        }
        hnswlib::InnerProductSpace space(dim);
        const float expected = hnswlib::InnerProductDistance(a.data(), b.data(), &dim);
        const float actual = space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
        if (std::abs(expected - actual) > 0.00001f) {
            std::cerr << "Native inner product differs from scalar distance at dimension " << dim << '\n';
            return EXIT_FAILURE;
        }
        hnswlib::L2Space l2_space(dim);
        const float expected_l2 = hnswlib::L2Sqr(a.data(), b.data(), &dim);
        const float actual_l2 = l2_space.get_dist_func()(a.data(), b.data(), l2_space.get_dist_func_param());
        if (std::abs(expected_l2 - actual_l2) > 0.00001f) {
            std::cerr << "Native squared L2 differs from scalar distance at dimension " << dim << '\n';
            return EXIT_FAILURE;
        }
    }
    // A BF16 vector crossing two SIMD blocks and a scalar tail must not leave
    // holes. Binary fractions are exactly representable by both encoders.
    std::vector<float> bf16_source(33);
    for (size_t i = 0; i < bf16_source.size(); ++i) bf16_source[i] = float(i + 1) / 4.0f;
    std::vector<uint8_t> bf16_code(bf16_source.size() * sizeof(uint16_t));
    hnswlib::IntStorage::quantize(hnswlib::StorageType::BF16, bf16_source.data(),
                                bf16_code.data(), bf16_source.size());
    for (size_t i = 0; i < bf16_source.size(); ++i) {
        uint16_t code;
        std::memcpy(&code, bf16_code.data() + i * sizeof(code), sizeof(code));
        uint32_t float_bits;
        std::memcpy(&float_bits, &bf16_source[i], sizeof(float_bits));
        if (code != uint16_t(float_bits >> 16)) {
            std::cerr << "BF16 packing skipped a component\n";
            return EXIT_FAILURE;
        }
    }
    storage.clear();
    std::cout << "Header layout regression passed\n";
}
