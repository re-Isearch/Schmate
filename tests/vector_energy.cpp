#include <atomic>
#include <cstring>
#include <functional>
#include "unified_hnsw.hpp"
#include <cstdlib>
#include <future>
#include <iostream>
#include <limits>
#include <map>

using namespace hnswlib;

static void require(bool condition, const char* message)
{
    if (!condition) { std::cerr << message << '\n'; std::exit(1); }
}

static bool close(float a, float b) { return std::abs(a - b) < 0.0003f; }

static std::vector<uint8_t> pack(const std::vector<int>& values, int bits)
{
    std::vector<uint8_t> code((values.size() * bits + 7) / 8, 0);
    const unsigned mask = (1u << bits) - 1;
    for (size_t i = 0; i < values.size(); ++i)
        for (int bit = 0; bit < bits; ++bit)
            if ((unsigned(values[i]) & mask) & (1u << bit))
                code[(i * bits + bit) / 8] |= uint8_t(1u << ((i * bits + bit) % 8));
    return code;
}

static void decoder_regressions()
{
    for (auto storage : {StorageType::INT2, StorageType::INT3, StorageType::INT4,
                         StorageType::INT5, StorageType::INT6, StorageType::INT8,
                         StorageType::INT16}) {
        const int bits = IntStorage::bits_per_element(storage);
        const int low = -(1 << (bits - 1));
        const std::vector<int> values{low, 1, -1, 0, low + 1};
        const auto code = pack(values, bits);
        SpaceQuantizedIP<float> space(values.size(), storage, QuantMode::INT4, OptBinMode::PASS);
        double expected = 0;
        for (int value : values) expected += double(value) * value;
        require(space.supports_query_energy(), "signed PASS support");
        require(space.query_energy(code.data()) == expected, "energy uses signed packed decoder");
        const float native = space.get_dist_func()(code.data(), code.data(), space.get_dist_func_param());
        require(close(native, float(1.0 - expected / space.reference_energy)), "native metric unchanged");
    }
    SpaceQuantizedIP<float> binary(5, StorageType::BIN1, QuantMode::BIN1, OptBinMode::PASS);
    require(!binary.supports_query_energy(), "binary PASS is not signed integer calibration");
    SpaceQuantizedIP<float> standard(5, StorageType::INT4, QuantMode::INT4, OptBinMode::STANDARD);
    require(!standard.supports_query_energy(), "non-PASS quantizer not reinterpreted as signed codes");
}

struct EvenLabels : BaseFilterFunctor {
    bool operator()(labeltype label) override { return label % 2 == 0; }
};

static void search_regressions(Metric metric, QuantMode quantization, StorageType storage)
{
    constexpr size_t dim = 65, count = 7;
    UnifiedIndexMeta meta(dim, 32, metric, quantization, OptBinMode::PASS, storage, false, 4, 40);
    meta.storage_mode_ = VectorStorageMode::DISABLED;
    UnifiedIndex index(meta);
    SpaceQuantizedIP<float> space(dim, storage, quantization, OptBinMode::PASS);
    HierarchicalNSW<float> native(&space, 32, 4, 40);
    std::vector<std::vector<float>> vectors(count, std::vector<float>(dim));
    for (size_t row = 0; row < count; ++row) {
        for (size_t column = 0; column < dim; ++column)
            vectors[row][column] = (storage == StorageType::INT8 ? 4.0f : 0.08f) *
                                  std::sin(float((row + 1) * (column + 1)));
        if (metric == Metric::Cosine) normalize_l2(vectors[row].data(), dim);
        index.addPoint(vectors[row].data(), row);
        std::vector<uint8_t> code(space.get_data_size());
        space.quantize(vectors[row].data(), code.data());
        native.addPoint(code.data(), row);
    }
    index.setEf(32); native.setEf(32);
    for (const auto& query : vectors) {
        std::vector<uint8_t> code(space.get_data_size());
        space.quantize(query.data(), code.data());
        const double energy = space.query_energy(code.data());
        const auto raw = native.searchKnnCloserFirst(code.data(), count);
        const auto calibrated = index.searchKnnCloserFirst(query.data(), count, false);
        require(raw.size() == calibrated.size(), "same candidates");
        auto heap = index.searchKnn(query.data(), count, false);
        std::map<labeltype, float> heap_distances;
        while (!heap.empty()) { heap_distances[heap.top().second] = heap.top().first; heap.pop(); }
        for (size_t i = 0; i < raw.size(); ++i) {
            require(raw[i].second == calibrated[i].second, "ANN order unchanged");
            const float expected = energy > 0 ?
                float(1.0 - space.reference_energy * (1.0 - raw[i].first) / energy) : 1.0f;
            if (!close(calibrated[i].first, expected))
                std::cerr << "metric=" << int(metric) << " storage=" << int(storage)
                          << " label=" << raw[i].second << " energy=" << energy
                          << " raw=" << raw[i].first << " expected=" << expected
                          << " actual=" << calibrated[i].first << '\n';
            require(close(calibrated[i].first, expected), "dot divided by query energy");
            require(close(heap_distances.at(calibrated[i].second), calibrated[i].first), "heap/ordered parity");
        }
        EvenLabels filter;
        for (const auto& hit : index.searchKnnCloserFirst(query.data(), count, &filter, false))
            require(hit.second % 2 == 0, "filtered search preserves filter");
    }
    std::vector<float> zero(dim, 0);
    for (const auto& hit : index.searchKnnCloserFirst(zero.data(), count, false))
        require(hit.first == 1.0f, "zero-energy neutral distance");
    for (const auto& hit : index.searchWithStopCondition(zero.data(), 0, 1, count))
        require(hit.first == 1.0f, "stop-condition energy normalization");
    const auto self = index.searchKnnCloserFirst(vectors[0].data(), count, false);
    const auto stop = index.searchWithStopCondition(vectors[0].data(), 10, count, count);
    require(stop.size() == count, "stop-condition returns calibrated candidates");
    for (const auto& hit : stop) {
        const auto found = std::find_if(self.begin(), self.end(), [&](const auto& x) { return x.second == hit.second; });
        require(found != self.end() && close(found->first, hit.first), "stop/ordered parity");
    }
    std::vector<std::future<void>> threads;
    for (size_t thread = 0; thread < 8; ++thread)
        threads.push_back(std::async(std::launch::async, [&, thread] {
            const auto& query = vectors[thread % count];
            const auto expected = index.searchKnnCloserFirst(query.data(), count, false);
            for (int trial = 0; trial < 50; ++trial) {
                const auto actual = index.searchKnnCloserFirst(query.data(), count, false);
                require(actual == expected, "concurrent query isolation");
            }
        }));
    for (auto& thread : threads) thread.get();
    require(index.score_from_dist(0) == 1 && index.score_from_dist(1) == 0.5f &&
            index.score_from_dist(2) == 0, "source score convention");
    require(index.score_from_dist(-1) == 1 && index.score_from_dist(3) == 0, "source score bounded");
}

static void rescore_regressions()
{
    UnifiedIndexMeta meta(3, 16, Metric::Cosine, QuantMode::INT8, OptBinMode::STANDARD,
                          StorageType::INT8, true, 4, 40);
    meta.storage_mode_ = VectorStorageMode::IN_MEMORY;
    UnifiedIndex index(meta);
    const float query[]{1, 0, 0}, opposite[]{-1, 0, 0}, orthogonal[]{0, 1, 0};
    index.addPoint(query, 0); index.addPoint(opposite, 1); index.addPoint(orthogonal, 2);
    const auto ordered = index.searchKnnCloserFirst(query, 3, true);
    require(ordered.size() == 3 && close(ordered[0].first, 0) &&
            close(ordered[1].first, 1) && close(ordered[2].first, 2), "rescoring uses 1 - similarity");
    auto heap = index.searchKnn(query, 3, true);
    require(heap.size() == 3 && close(heap.top().first, 2), "rescored heap convention");
    const auto stop = index.searchWithStopCondition(query, 0.1f, 1, 3);
    require(stop.size() == 1 && close(stop[0].first, 0), "rescored stop convention");
    UnifiedIndexMeta l2meta(3, 16, Metric::L2, QuantMode::INT4, OptBinMode::PASS, StorageType::INT4);
    l2meta.storage_mode_ = VectorStorageMode::DISABLED;
    UnifiedIndex l2(l2meta);
    l2.addPoint(query, 0);
    require(close(l2.searchKnnCloserFirst(query, 1, false)[0].first, 0), "L2 remains unchanged");
}

int main()
{
    decoder_regressions();
    search_regressions(Metric::IP, QuantMode::INT4, StorageType::INT4);
    search_regressions(Metric::Cosine, QuantMode::INT4, StorageType::INT4);
    search_regressions(Metric::IP, QuantMode::INT8, StorageType::INT8);
    rescore_regressions();
    std::cout << "Vector energy regressions passed\n";
}
