// Sample the same query-energy-normalized score that leaves signed PASS IP
// search, using a row-major float32 embedding export from one model and field.
#include <atomic>
#include <cstring>
#include <functional>
#include "unified_hnsw.hpp"
#include <iomanip>
#include <random>

using namespace hnswlib;

int main(int argc, char** argv)
try {
    if (argc < 3 || argc > 5) {
        std::cerr << "Usage: vector_background DIM embeddings.f32 [PAIRS=100000] [SEED=42]\n";
        return 1;
    }
    const size_t dim = std::stoull(argv[1]);
    const size_t pairs = argc > 3 ? std::stoull(argv[3]) : 100000;
    const unsigned seed = argc > 4 ? std::stoul(argv[4]) : 42;
    if (!dim || dim > 1000000 || !pairs) throw std::runtime_error("invalid dimensions or pair count");
    std::ifstream input(argv[2], std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open embedding export");
    const auto bytes = input.tellg();
    const size_t row_bytes = dim * sizeof(float);
    if (bytes < 0 || size_t(bytes) % row_bytes || size_t(bytes) / row_bytes < 2)
        throw std::runtime_error("need at least two complete float32 embedding rows");
    const size_t rows = size_t(bytes) / row_bytes;
    input.seekg(0);
    SpaceQuantizedIP<float> space(dim, StorageType::INT4, QuantMode::INT4, OptBinMode::PASS);
    std::vector<std::vector<uint8_t>> codes(rows, std::vector<uint8_t>(space.get_data_size()));
    std::vector<double> energies(rows);
    std::vector<float> vector(dim);
    double sum = 0, sum_squares = 0, minimum = std::numeric_limits<double>::max(), maximum = 0;
    size_t zero_count = 0;
    for (size_t row = 0; row < rows; ++row) {
        if (!input.read(reinterpret_cast<char*>(vector.data()), row_bytes))
            throw std::runtime_error("truncated embedding export");
        for (float component : vector)
            if (!std::isfinite(component)) throw std::runtime_error("non-finite embedding component");
        normalize_l2(vector.data(), dim);
        space.quantize(vector.data(), codes[row].data());
        const double energy = space.query_energy(codes[row].data());
        energies[row] = energy;
        sum += energy; sum_squares += energy * energy;
        minimum = std::min(minimum, energy); maximum = std::max(maximum, energy);
        if (energy == 0) ++zero_count;
    }
    const double mean = sum / rows;
    const double deviation = std::sqrt(std::max(0.0, sum_squares / rows - mean * mean));
    std::cerr << "rows=" << rows << " mean_energy=" << mean << " stddev=" << deviation
              << " coefficient_of_variation=" << (mean > 0 ? deviation / mean : 0)
              << " min=" << minimum << " max=" << maximum << " zero=" << zero_count << '\n';
    std::mt19937 generator(seed);
    std::uniform_int_distribution<size_t> row_distribution(0, rows - 1);
    const auto distance = space.get_dist_func();
    std::cout << std::setprecision(9);
    for (size_t sample = 0; sample < pairs; ++sample) {
        const size_t query = row_distribution(generator);
        size_t candidate;
        do { candidate = row_distribution(generator); } while (candidate == query);
        const double native = distance(codes[query].data(), codes[candidate].data(),
                                       space.get_dist_func_param());
        const double similarity = energies[query] > 0 ?
            space.reference_energy * (1.0 - native) / energies[query] : 0.0;
        std::cout << std::clamp((1.0 + similarity) / 2.0, 0.0, 1.0) << '\n';
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "vector_background: " << error.what() << '\n';
    return 1;
}
