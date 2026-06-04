#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <filesystem>

#define HNSWLIB_NO_SYSTEM_SAFETY_CHECK
#define HNSWERR std::cerr

#include "LSMVectorStorage.h"

using namespace hnswlib;

// ------------------------------------------
// Utility: create deterministic random vector
// ------------------------------------------
std::vector<float> make_vec(size_t dim, int seed) {
    std::vector<float> v(dim);
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    for (size_t i = 0; i < dim; i++) v[i] = dist(g);
    return v;
}

// ------------------------------------------
// Utility: write a synthetic "unified index" file with vectors
// Format:
//   [size_t num_vectors]
//   [(labeltype)(float * dim)] * num_vectors
// ------------------------------------------
void write_unified_index(
    const std::string &fname,
    const std::vector<std::pair<labeltype, std::vector<float>>> &items)
{
    std::ofstream ofs(fname, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        HNSWERR << "ERROR: LSMStorage failed to write unified-file.\n";
        exit(1);
    }

    size_t n = items.size();
    ofs.write((char*)&n, sizeof(n));

    for (auto &p : items) {
        labeltype label = p.first;
        ofs.write((char*)&label, sizeof(label));
        ofs.write((char*)p.second.data(), p.second.size() * sizeof(float));
std::cerr << "Writing [" << label << "] = " << p.second.data()[0] << std::endl;
    }
    ofs.close();
    std::cout << "[Write unified index OK]\n";
}

// ------------------------------------------
// Utility: verify vector equality
// ------------------------------------------
bool vec_equal(const float *a, const std::vector<float> &b, size_t dim) {
    for (size_t i = 0; i < dim; i++) {
        if (fabs(a[i] - b[i]) > 1e-6)
            return false;
    }
    return true;
}

// ------------------------------------------
// End-to-end test for streaming compaction
// ------------------------------------------
void test_streaming_compaction() {
    std::cout << "\n===== TEST: STREAMING COMPACTION =====\n";

    const size_t dim = 5;
    const std::string base = "tmp_test_index.bin";
    const std::string clean_glob = "tmp_test_index.bin*";

    // cleanup old files
    for (auto &f : std::filesystem::directory_iterator(".")) {
        std::string s = f.path().string();
        if (s.rfind("tmp_test_index", 0) == 0) {
            std::filesystem::remove(s);
        }
    }

   auto val10 = make_vec(dim, 10);
   auto val20 = make_vec(dim, 20);
   auto val30 = make_vec(dim, 30);


    // 1) Write unified index with initial items
    std::vector<std::pair<labeltype, std::vector<float>>> initial = {
        {10, val10},
        {20, val20},
        {30, val30}
    };

std::cerr << "E 10 = " << val10[0] << std::endl;;
std::cerr << "E 20 = " << val20[0] << std::endl;;
std::cerr << "E 30 = " << val30[0] << std::endl;;


    write_unified_index(base, initial);

    // 2) Load using LSMVectorStorage
    LSMVectorStorage store(dim);
    VectorStorageConfig cfg;
    cfg.mode = VectorStorageMode::MEMORY_MAPPED;
    cfg.use_streaming_compaction = true;
    store.set_config(cfg);

    {
        std::ifstream ifs(base, std::ios::binary);
        assert(ifs.good());
        store.load_vectors(base, ifs, VectorStorageMode::MEMORY_MAPPED, 0);
    }

    // 3) Validate initial reads


std::cerr << "10 = " << store.get_vector(10)[0] << std::endl;;
std::cerr << "20 = " << store.get_vector(20)[0] << std::endl;;
std::cerr << "30 = " << store.get_vector(30)[0] << std::endl;;

    for (auto &p : initial) {
std::cerr << "P.FIRST = " << p.first << std::endl;
std::cerr << "P.SECOND = " << (p.second)[0] << std::endl;
        const float *vec = store.get_vector(p.first);
std::cerr << "VEC = " << vec[0] << std::endl;

        assert(vec && vec_equal(vec, p.second, dim));
    }
    std::cout << "[Initial load OK]\n";

    // 4) Add delta updates
    // overwrite label 20, add new label 40
    store.addPoint(20, make_vec(dim, 999).data());  // new data for label 20
    store.addPoint(40, make_vec(dim, 40).data());
    store.flush_delta();

    // 5) Run streaming compaction
    bool ok = store.compact();
    assert(ok);
    std::cout << "[Streaming compaction OK]\n";

    // 6) Validate post-compaction reads
    {
        // label 10 and 30 should be unchanged from initial
        auto p10 = initial[0];
        auto p30 = initial[2];
        assert(vec_equal(store.get_vector(10), p10.second, dim));
        assert(vec_equal(store.get_vector(30), p30.second, dim));

        // label 20 must be the UPDATED vector we wrote
        std::vector<float> updated20 = make_vec(dim, 999);
        assert(vec_equal(store.get_vector(20), updated20, dim));

        // label 40 must exist and match
        std::vector<float> v40 = make_vec(dim, 40);
        assert(vec_equal(store.get_vector(40), v40, dim));
    }

    std::cout << "[Post-compaction vector verification OK]\n";

    // 7) Validate that only *one* compact file exists
    int count = 0;
    for (auto &f : std::filesystem::directory_iterator(".")) {
	const char name[] = "tmp_test_index.bin.vectors.";
        std::string path_str = f.path().string();
        std::string fname = std::filesystem::path(path_str).filename().string();
        if (fname.rfind(name, 0) == 0) {
            count++;
            std::cout << "Found compact file: " << fname << "\n";
        }
    }

    assert(count == 1);

    std::cout << "[Compact file count OK]\n";

    // 8) Validate mmap reload
    {
        LSMVectorStorage store2(dim);
        store2.set_config(cfg);

        // discover it again
        bool ok = store2.load_vectors(base, *(new std::ifstream(base, std::ios::binary)),
                            VectorStorageMode::MEMORY_MAPPED);
        auto v20 = make_vec(dim, 999); // was 20
std::cerr << "v20 = " << v20[0] << std::endl;
std::cerr << "[20] = " << store2.get_vector(20)[0] << std::endl;
        assert(vec_equal(store2.get_vector(20), v20, dim));

        std::cout << "[Reload after compact OK]\n";
    }

    std::cout << "===== STREAMING COMPACTION TEST PASSED 🎉 =====\n\n";
}

// ------------------------------------------
// MAIN
// ------------------------------------------
int main() {
    test_streaming_compaction();

    std::cout << "ALL TESTS PASSED\n";
    return 0;
}

