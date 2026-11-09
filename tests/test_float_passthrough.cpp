#include "unified_hnsw.hpp"
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace hnswlib;

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static void near(double actual, double expected, double tolerance, const char* message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::cerr << message << ": actual=" << actual << " expected=" << expected << '\n';
        throw std::runtime_error(message);
    }
}

static std::vector<uint8_t> encode(SpaceInterface<float>& space, const std::vector<float>& values) {
    std::vector<uint8_t> bytes(space.get_data_size());
    space.quantize(values.data(), bytes.data());
    return bytes;
}

static float distance(SpaceInterface<float>& space, const uint8_t* a, const uint8_t* b) {
    return space.get_dist_func()(a, b, space.get_dist_func_param());
}

static void native_distances(StorageType storage, size_t dim) {
    SpaceQuantizedIP<float> ip(dim, storage, QuantMode::NONE, OptBinMode::PASS);
    SpaceQuantized<float> l2(dim, storage, QuantMode::NONE, OptBinMode::PASS);
    require(!ip.supports_query_energy(), "floating PASS must not use signed-integer query-energy calibration");
    std::vector<float> a(dim), b(dim);
    double dot = 0, squared = 0;
    // Binary fractions are exactly representable in all three formats.
    for (size_t i = 0; i < dim; ++i) {
        a[i] = float(int(i % 7) - 3) / 32;
        b[i] = float(int(i % 5) - 2) / 16;
        dot += double(a[i]) * b[i];
        squared += double(a[i] - b[i]) * (a[i] - b[i]);
    }
    const auto qa = encode(ip, a), qb = encode(ip, b);
    near(distance(ip, qa.data(), qb.data()), 1 - dot, 1e-6,
         "floating IP decodes values rather than signed bit patterns or integer energy");
    near(distance(l2, qa.data(), qb.data()), squared, 1e-6,
         "floating L2 decodes values rather than nibbles");
    near(distance(l2, qa.data(), qa.data()), 0, 0, "floating L2 self distance is zero");
    std::vector<uint8_t> unaligned_a(qa.size() + 1), unaligned_b(qb.size() + 1);
    std::copy(qa.begin(), qa.end(), unaligned_a.begin() + 1);
    std::copy(qb.begin(), qb.end(), unaligned_b.begin() + 1);
    near(distance(ip, unaligned_a.data() + 1, unaligned_b.data() + 1), 1 - dot, 1e-6,
         "floating IP accepts packed unaligned vector addresses");
    near(distance(l2, unaligned_a.data() + 1, unaligned_b.data() + 1), squared, 1e-6,
         "floating L2 accepts packed unaligned vector addresses");
    b = a;
    b.back() += 1.0f / 32;
    const auto last = encode(l2, b);
    near(distance(l2, qa.data(), last.data()), 1.0 / 1024, 1e-9,
         "floating L2 reads the final component at odd and SIMD-tail dimensions");
}

static void half_values() {
    const uint16_t one = 0x3c00;
    const uint16_t codes[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x0400,
                              0x3800, 0xb800, 0x3c00, 0xbc00, 0x7bff};
    const double expected[] = {0, 0, std::ldexp(1.0, -24), -std::ldexp(1.0, -24),
                               std::ldexp(1023.0, -24), std::ldexp(1.0, -14),
                               0.5, -0.5, 1, -1, 65504};
    for (size_t i = 0; i < std::size(codes); ++i)
        near(IntStorage::floating_inner_product(StorageType::FP16,
             reinterpret_cast<const uint8_t*>(&codes[i]), reinterpret_cast<const uint8_t*>(&one), 1),
             expected[i], 0, "FP16 decoder handles signs, zero, subnormals and normal exponents");
}

static void raw_ip_units(StorageType storage) {
    UnifiedIndexMeta meta(3, 8, Metric::IP, QuantMode::NONE, OptBinMode::PASS, storage);
    meta.storage_mode_ = VectorStorageMode::DISABLED;
    UnifiedIndex index(meta);
    const float query[] = {0.25f, -0.25f, 0}, twice[] = {0.5f, -0.5f, 0};
    index.addPoint(query, 10); index.addPoint(twice, 20);
    const auto hits = index.searchKnnCloserFirst(query, 2, false);
    require(hits.size() == 2 && hits[0].second == 20, "floating IP preserves dot-product ordering");
    near(hits[0].first, 0.75, 1e-6, "floating IP uses original value units without signed query energy");
    near(hits[1].first, 0.875, 1e-6, "floating IP self distance retains its natural magnitude");
}

static void packed_row_alignment(StorageType storage, const std::filesystem::path& directory) {
    constexpr size_t dim = 3;
    SpaceQuantizedIP<float> space(dim, storage, QuantMode::NONE, OptBinMode::PASS);
    HierarchicalNSW<float> index(&space, 4, 4, 40);
    for (size_t row = 0; row < 4; ++row) {
        const auto code = encode(space, {float(row + 1) / 16, -0.125f, 0.25f});
        index.addPoint(code.data(), row);
        require(reinterpret_cast<uintptr_t>(index.get_linklist0(row)) % alignof(unsigned int) == 0,
                "odd-sized packed rows preserve link-list alignment");
    }
    const auto path = directory / ("aligned-" + std::to_string(int(storage)) + ".hix");
    index.saveIndex(path.string());
    HierarchicalNSW<float> loaded(&space, path.string());
    for (size_t row = 0; row < 4; ++row)
        require(reinterpret_cast<uintptr_t>(loaded.get_linklist0(row)) % alignof(unsigned int) == 0,
                "packed link-list alignment survives serialization");
}

struct EvenLabels : BaseFilterFunctor {
    bool operator()(labeltype label) override { return label % 2 == 0; }
};

static void search_roundtrip(StorageType storage, Metric metric, size_t dim,
                             const std::filesystem::path& directory) {
    constexpr size_t count = 16;
    std::mt19937 generator(1729);
    std::normal_distribution<float> normal;
    std::vector<std::vector<float>> vectors(count, std::vector<float>(dim));
    for (auto& vector : vectors) {
        for (auto& value : vector) value = normal(generator);
        normalize_l2(vector.data(), dim);
    }
    UnifiedIndexMeta meta(dim, count, metric, QuantMode::NONE, OptBinMode::PASS, storage, false, 8, 80);
    meta.storage_mode_ = VectorStorageMode::DISABLED;
    UnifiedIndex index(meta);
    for (size_t row = 0; row < count; ++row) index.addPoint(vectors[row].data(), row);
    index.setEf(count);
    const double tolerance = storage == StorageType::BF16 ? 0.03 :
                             storage == StorageType::FP16 ? 0.004 : 0.00002;
    auto expected = [&](size_t query, size_t candidate) {
        double value = 0;
        for (size_t i = 0; i < dim; ++i) {
            const double a = vectors[query][i], b = vectors[candidate][i];
            value += metric == Metric::L2 ? (a - b) * (a - b) : a * b;
        }
        return metric == Metric::L2 ? value : 1 - value;
    };
    auto verify = [&](UnifiedIndex& current) {
        std::vector<std::vector<std::pair<float, labeltype>>> all;
        for (size_t query : {size_t(0), size_t(3), size_t(8), size_t(15)}) {
            auto input = vectors[query];
            if (metric == Metric::Cosine)
                for (auto& value : input) value *= 3; // Production must normalize cosine queries.
            const auto hits = current.searchKnnCloserFirst(input.data(), count, false);
            require(hits.size() == count && hits.front().second == query,
                    "floating PASS ranks the matching sentence first");
            require(current.score_from_dist(hits.front().first) > 0.99,
                    "floating PASS gives a high score to the matching normalized vector");
            for (const auto& hit : hits) {
                near(hit.first, expected(query, hit.second), tolerance,
                     "floating search distance matches the float reference");
                if (hit.second != query)
                    require(current.score_from_dist(hit.first) < 0.8,
                            "unrelated floating PASS vectors must not all saturate to score one");
            }
            auto heap = current.searchKnn(input.data(), count, false);
            require(heap.size() == count, "floating heap and ordered searches return the same candidates");
            while (!heap.empty()) {
                const auto hit = heap.top(); heap.pop();
                near(hit.first, expected(query, hit.second), tolerance, "floating heap distance units agree");
            }
            EvenLabels filter;
            const auto filtered = current.searchKnnCloserFirst(input.data(), count, &filter, false);
            require(filtered.size() == count / 2, "floating filtered search returns every permitted label");
            for (const auto& hit : filtered) {
                require(hit.second % 2 == 0, "floating search preserves its filter");
                near(hit.first, expected(query, hit.second), tolerance, "floating filtered distances agree");
            }
            const auto epsilon = current.searchWithStopCondition(input.data(), 0.03f, 1, count);
            require(epsilon.size() == 1 && epsilon.front().second == query,
                    "floating epsilon search selects the matching vector in natural distance units");
            all.push_back(hits);
        }
        return all;
    };
    const auto before = verify(index);
    const auto path = directory / (std::to_string(int(storage)) + "-" +
                                   std::to_string(int(metric)) + "-" + std::to_string(dim) + ".hix");
    require(index.saveIndex(path.string()), "save floating PASS index");
    UnifiedIndex loaded(meta);
    require(loaded.loadIndex(path.string()), "reload floating PASS index");
    loaded.setEf(count);
    require(verify(loaded) == before, "floating PASS ranking and distances survive save/reload");
}

int main() {
    const auto directory = std::filesystem::current_path() / "float-passthrough-fixtures";
    bool created = false;
    try {
        created = std::filesystem::create_directory(directory);
        require(created, "create an isolated floating regression directory");
        half_values();
        for (auto storage : {StorageType::FP16, StorageType::BF16, StorageType::INT8})
            packed_row_alignment(storage, directory);
        for (auto storage : {StorageType::FP16, StorageType::BF16, StorageType::FLOAT32}) {
            for (size_t dim : {1, 3, 31, 32, 33, 127, 384, 768, 1024}) native_distances(storage, dim);
            raw_ip_units(storage);
            for (auto metric : {Metric::Cosine, Metric::IP, Metric::L2})
                for (size_t dim : {768, 1024}) search_roundtrip(storage, metric, dim, directory);
        }
        std::filesystem::remove_all(directory);
        std::cout << "Floating PASS decoding, distances, search and persistence: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        if (created) std::filesystem::remove_all(directory);
        std::cerr << "Floating PASS regression: " << error.what() << '\n';
        return 1;
    }
}
