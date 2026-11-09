#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "unified_hnsw.hpp"

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

static std::vector<uint8_t> encode(SpaceInterface<float>& space, const std::vector<float>& input) {
    std::vector<uint8_t> bytes(space.get_data_size());
    space.quantize(input.data(), bytes.data());
    return bytes;
}

static float distance(SpaceInterface<float>& space, const std::vector<uint8_t>& a,
                      const std::vector<uint8_t>& b) {
    return space.get_dist_func()(a.data(), b.data(), space.get_dist_func_param());
}

static void signed_encoding() {
    const std::vector<float> values = {0.03125f, -0.03125f, 0.0625f, -0.0625f,
                                      0.0f, 1.0f, -1.0f, 3.0f, -3.0f};
    const int expected[] = {4, -4, 8, -8, 0, 127, -127, 127, -127};
    std::vector<uint8_t> bytes(values.size() + 8, 0xa5);
    IntStorage::quantize(StorageType::INT8, values.data(), bytes.data(), values.size());
    for (size_t i = 0; i < values.size(); ++i)
        require(int(int8_t(bytes[i])) == expected[i], "signed Q8 retains small positive and negative components");
    require(std::all_of(bytes.begin() + values.size(), bytes.end(),
                        [](uint8_t value) { return value == 0xa5; }), "Q8 writes one byte per component");
    std::vector<double> doubles(values.begin(), values.end());
    std::vector<uint8_t> from_double(values.size());
    IntStorage::quantize(StorageType::INT8, doubles.data(), from_double.data(), doubles.size());
    require(std::equal(from_double.begin(), from_double.end(), bytes.begin()), "float and double Q8 input agree");
}

static double coded_dot(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    double dot = 0;
    for (size_t i = 0; i < a.size(); ++i) dot += int(int8_t(a[i])) * int(int8_t(b[i]));
    return dot;
}

static void native_energy_separation(size_t dim) {
    SpaceQuantizedIP<float> space(dim, StorageType::INT8, QuantMode::NONE, OptBinMode::PASS);
    require(space.get_data_size() == dim, "Q8 layout remains one byte per dimension");
    const float component = 1.0f / std::sqrt(float(dim));
    std::vector<float> a(dim, component), b(dim, -component), mixed = a;
    for (size_t i = 1; i < dim; i += 2) mixed[i] = -component;
    const auto qa = encode(space, a), qb = encode(space, b), qm = encode(space, mixed);
    require(std::any_of(qa.begin(), qa.end(), [](uint8_t v) { return v != 0; }),
            "normalized 768D/1024D embeddings must not collapse to zero");
    const double energy = coded_dot(qa, qa);
    near(space.query_energy(qa.data()), energy, 0.0, "Q8 energy uses signed stored components");
    for (const auto* candidate : {&qa, &qb, &qm})
        near(distance(space, qa, *candidate),
             1.0 - coded_dot(qa, *candidate) / space.reference_energy, 1e-6,
             "native Q8 distance keeps the fixed traversal denominator");
    std::vector<uint8_t> zero(dim, 0);
    near(space.query_energy(zero.data()), 0.0, 0.0, "zero-vector energy stays zero");
    near(distance(space, zero, qa), 1.0, 1e-6, "zero-vector native distance stays finite");
}

static void metric_units() {
    const std::vector<float> a = {0.25f, -0.5f, 0.03125f, -0.0625f, 0.125f};
    const std::vector<float> b = {-0.5f, 0.25f, -0.0625f, 0.03125f, 0.25f};
    SpaceQuantizedIP<float> ip(a.size(), StorageType::INT8, QuantMode::NONE, OptBinMode::PASS);
    SpaceQuantized<float> l2(a.size(), StorageType::INT8, QuantMode::NONE, OptBinMode::PASS);
    const auto qa = encode(ip, a), qb = encode(ip, b);
    double squared = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double da = double(int8_t(qa[i])) / 127.0;
        const double db = double(int8_t(qb[i])) / 127.0;
        squared += (da - db) * (da - db);
    }
    near(distance(ip, qa, qb), 1.0 - coded_dot(qa, qb) / ip.reference_energy, 1e-6,
         "native Q8 IP retains the query-energy branch traversal metric");
    near(distance(l2, qa, qb), squared, 1e-6, "Q8 L2 uses signed bytes and original vector units");
    near(distance(l2, qa, qa), 0.0, 1e-6, "Q8 L2 self distance is zero");
    // The old four-bit dispatch ignored this part of an INT8 vector.
    auto changed = qa;
    changed.back() = uint8_t(int8_t(int8_t(changed.back()) + 1));
    near(distance(l2, qa, changed), 1.0 / (127.0 * 127.0), 1e-7,
         "Q8 L2 reads the final byte of an odd-dimensional vector");
}

static void other_encodings() {
    const float values[] = {0.03125f, -0.03125f, 0.0625f, -0.0625f};
    uint8_t q4[2] = {};
    IntStorage::quantize(StorageType::INT4, values, q4, 4);
    require(q4[0] == 0xe2 && q4[1] == 0xc4, "Q4 encoding remains compatible");
    const float halves[] = {0.5f, -0.5f, 1.0f, -1.0f};
    uint16_t fp16[4] = {};
    IntStorage::quantize(StorageType::FP16, halves, reinterpret_cast<uint8_t*>(fp16), 4);
    require(fp16[0] == 0x3800 && fp16[1] == 0xb800 && fp16[2] == 0x3c00 && fp16[3] == 0xbc00,
            "FP16 encoding remains compatible");
    float fp32[4] = {};
    IntStorage::quantize(StorageType::FLOAT32, halves, reinterpret_cast<uint8_t*>(fp32), 4);
    require(std::memcmp(fp32, halves, sizeof(halves)) == 0, "FP32 pass-through remains exact");
}

static void query_energy_contract() {
    UnifiedIndexMeta meta(3, 8, Metric::IP, QuantMode::NONE, OptBinMode::PASS, StorageType::INT8);
    meta.storage_mode_ = VectorStorageMode::DISABLED;
    UnifiedIndex index(meta);
    const float query[] = {0.25f, -0.25f, 0};
    const float twice[] = {0.5f, -0.5f, 0};
    index.addPoint(query, 10);
    index.addPoint(twice, 20);
    const auto result = index.searchKnnCloserFirst(query, 2, false);
    require(result.size() == 2 && result[0].second == 20 && result[1].second == 10,
            "query energy preserves candidate magnitude instead of using full cosine");
    near(result[0].first, -1.0, 1e-6, "twice the query has similarity two in query-energy units");
    near(result[1].first, 0.0, 1e-6, "self similarity is one in query-energy units");
    near(index.score_from_dist(result[0].first), 1.0, 1e-6, "public scores bound amplified similarity");
    const float zero[] = {0, 0, 0};
    for (const auto& hit : index.searchKnnCloserFirst(zero, 2, false))
        near(index.score_from_dist(hit.first), 0.5, 1e-6, "zero-energy queries retain neutral source scores");
}

static void graph_roundtrip(Metric metric, size_t dim, const std::filesystem::path& directory) {
    SpaceQuantizedIP<float> space(dim, StorageType::INT8, QuantMode::NONE, OptBinMode::PASS);
    std::mt19937 generator(1729);
    std::normal_distribution<float> random;
    std::vector<std::vector<float>> vectors(32, std::vector<float>(dim));
    for (auto& vector : vectors) {
        double norm = 0.0;
        for (float& value : vector) { value = random(generator); norm += double(value) * value; }
        for (float& value : vector) value /= float(std::sqrt(norm));
    }
    std::vector<std::vector<uint8_t>> packed;
    UnifiedIndexMeta meta(dim, vectors.size(), metric, QuantMode::NONE,
                          OptBinMode::PASS, StorageType::INT8, false, 16, 100);
    meta.storage_mode_ = VectorStorageMode::DISABLED;
    UnifiedIndex index(meta);
    for (size_t i = 0; i < vectors.size(); ++i) {
        auto prepared = vectors[i];
        if (metric == Metric::Cosine) normalize_l2(prepared.data(), dim);
        packed.push_back(encode(space, prepared));
        index.addPoint(vectors[i].data(), i);
    }
    index.setEf(vectors.size());
    const auto verify = [&](UnifiedIndex& current) {
        for (size_t query : {size_t(0), size_t(7), size_t(19), size_t(31)}) {
            const double energy = coded_dot(packed[query], packed[query]);
            const auto result = current.searchKnnCloserFirst(vectors[query].data(), vectors.size(), false);
            require(result.size() == vectors.size() && result.front().second == query,
                    "Q8 search finds the matching vector rather than arbitrary tied labels");
            near(result.front().first, 0.0, 1e-6, "Q8 returned match distance is zero");
            near(current.score_from_dist(result.front().first), 1.0, 1e-6,
                 "matching Q8 source score is not the broken 0.5 constant");
            for (const auto& hit : result) {
                const size_t other = hit.second;
                const double expected = 1.0 - coded_dot(packed[query], packed[other]) / energy;
                near(hit.first, expected, 1e-6, "Q8 returned distance is calibrated exactly once by query energy");
                double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
                for (size_t d = 0; d < dim; ++d) {
                    const double a = vectors[query][d], b = vectors[other][d];
                    dot += a * b; norm_a += a * a; norm_b += b * b;
                }
                near(hit.first, 1.0 - dot / std::sqrt(norm_a * norm_b), 0.025,
                     "Q8 calibrated distances stay close to the normalized float reference");
            }
            auto heap = current.searchKnn(vectors[query].data(), vectors.size(), false);
            require(heap.size() == result.size(), "Q8 heap and ordered searches return the same candidates");
            while (!heap.empty()) {
                const auto hit = heap.top(); heap.pop();
                near(hit.first, 1.0 - coded_dot(packed[query], packed[hit.second]) / energy, 1e-6,
                     "Q8 heap search preserves query-energy calibration");
            }
            const auto stop = current.searchWithStopCondition(vectors[query].data(), 0.02f, 1, vectors.size());
            require(stop.size() == 1 && stop.front().second == query, "Q8 epsilon uses calibrated distance units");
            near(stop.front().first, result.front().first, 1e-6, "Q8 epsilon and ordered scores agree");
        }
    };
    verify(index);
    const std::string filename = (directory / ("index-" + std::to_string(int(metric)) + "-" +
                                              std::to_string(dim) + ".hix")).string();
    require(index.saveIndex(filename), "save Q8 index and quantization parameters");
    UnifiedIndex loaded(meta);
    require(loaded.loadIndex(filename), "reload Q8 index and quantization parameters");
    loaded.setEf(vectors.size());
    verify(loaded);
}

int main() {
    const auto directory = std::filesystem::current_path() / "q8-passthrough-fixtures";
    bool created = false;
    try {
        created = std::filesystem::create_directory(directory);
        require(created, "create an isolated regression directory");
        signed_encoding();
        for (size_t dim : {1, 3, 31, 32, 33, 127, 384, 768, 1024}) native_energy_separation(dim);
        metric_units();
        other_encodings();
        query_energy_contract();
        for (auto metric : {Metric::Cosine, Metric::IP}) {
            graph_roundtrip(metric, 768, directory);
            graph_roundtrip(metric, 1024, directory);
        }
        std::filesystem::remove_all(directory);
        std::cout << "Q8 pass-through encoding, metrics, ranking and persistence: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        if (created) std::filesystem::remove_all(directory);
        std::cerr << "Q8 pass-through regression: " << error.what() << '\n';
        return 1;
    }
}
