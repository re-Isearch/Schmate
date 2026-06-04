#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <cstring>
#include <algorithm>

enum class Metric {
    L2,
    InnerProduct,
    Cosine
};

enum class SearchMode {
    Knn,
    Radius,
    Relative,
    Adaptive,
    Epsilon
};

struct HnswConfig {
    // Search mode
    SearchMode default_search_mode = SearchMode::Knn;

    // Index parameters
    size_t max_elements = 100000;
    size_t M = 16;
    size_t ef_construction = 200;
    size_t ef_search = 50;
    Metric metric = Metric::Cosine;

    // BERT/Embedding
    size_t bert_n_threads = 4;

    // Chunking
    int max_tokens_per_chunk = 128;
    float overlap_percent = 0.1f;

    // Debug
    bool debug = true;

    // Search mode defaults
    size_t default_k = 5;          // knn
    float default_radius = 0.7f;   // radius: min score
    float default_alpha = 0.8f;    // relative: threshold multiplier
    size_t default_minN = 3;       // adaptive: minimum results
    size_t default_lookahead = 10; // adaptive: lookahead window
    float default_gapDelta = 0.1f; // adaptive: gap threshold

    // Epsilon search
    float default_epsilon = 0.15f;   // General epsilon (if < 0, use radius)
    float default_epsilonL2 = 1.41f; // L2 distance threshold (will be squared)
    float default_epsilonIP = 0.5f;  // InnerProduct distance threshold

    size_t min_candidates = 10;      // Min candidates for epsilon search
    size_t max_candidates_cap = 0;   // 0 = auto (no cap)

    // Performance tuning
    size_t knn_lookahead_scale = 5;
    int flush_threshold = 100;       // -1 = only explicit flush
    bool flush_offsets_each = false;

    // Parallel processing
    bool parallel_merge = true;
    unsigned merge_threads = 0;      // 0 = auto

    // Normalization
    bool normalized_embeddings = false;

    // Validation
    bool validate() const {
        if (max_elements == 0) {
            std::cerr << "Error: max_elements must be > 0\n";
            return false;
        }
        if (M == 0 || M > 128) {
            std::cerr << "Error: M must be in range [1, 128]\n";
            return false;
        }
        if (ef_construction < M * 2) {
            std::cerr << "Warning: ef_construction should be >= 2*M for good quality\n";
        }
        if (ef_search == 0) {
            std::cerr << "Error: ef_search must be > 0\n";
            return false;
        }
        if (overlap_percent < 0.0f || overlap_percent >= 1.0f) {
            std::cerr << "Error: overlap_percent must be in [0, 1)\n";
            return false;
        }
        if (max_tokens_per_chunk <= 0) {
            std::cerr << "Error: max_tokens_per_chunk must be > 0\n";
            return false;
        }
        if (default_k == 0) {
            std::cerr << "Error: default_k must be > 0\n";
            return false;
        }
        if (default_radius < 0.0f || default_radius > 1.0f) {
            std::cerr << "Warning: default_radius typically in [0, 1]\n";
        }
        return true;
    }

    // Get epsilon for current metric
    float get_epsilon() const {
        if (default_epsilon >= 0.0f) {
            return default_epsilon;
        }
        
        // Fall back to metric-specific epsilon
        switch (metric) {
            case Metric::L2:
                return default_epsilonL2 * default_epsilonL2; // Square it
            case Metric::InnerProduct:
            case Metric::Cosine:
                return default_epsilonIP;
            default:
                return 0.15f;
        }
    }

    // Get effective max_candidates (with cap applied)
    size_t get_max_candidates(size_t request = 0) const {
        if (request == 0) request = default_k * knn_lookahead_scale;
        if (max_candidates_cap > 0) {
            return std::min(request, max_candidates_cap);
        }
        return request;
    }

    // Get number of merge threads
    unsigned get_merge_threads() const {
        if (merge_threads == 0) {
            return std::thread::hardware_concurrency();
        }
        return merge_threads;
    }

    // Print configuration
    void print(std::ostream& os = std::cout) const {
        os << "=== HNSW Configuration ===\n";
        os << "Search mode: " << search_mode_to_string(default_search_mode) << "\n";
        os << "\nIndex parameters:\n";
        os << "  max_elements: " << max_elements << "\n";
        os << "  M: " << M << "\n";
        os << "  ef_construction: " << ef_construction << "\n";
        os << "  ef_search: " << ef_search << "\n";
        os << "  metric: " << metric_to_string(metric) << "\n";
        os << "  normalized_embeddings: " << (normalized_embeddings ? "yes" : "no") << "\n";
        
        os << "\nEmbedding:\n";
        os << "  bert_n_threads: " << bert_n_threads << "\n";
        
        os << "\nChunking:\n";
        os << "  max_tokens_per_chunk: " << max_tokens_per_chunk << "\n";
        os << "  overlap_percent: " << overlap_percent << "\n";
        
        os << "\nSearch defaults:\n";
        os << "  k (knn): " << default_k << "\n";
        os << "  radius: " << default_radius << "\n";
        os << "  alpha (relative): " << default_alpha << "\n";
        os << "  minN (adaptive): " << default_minN << "\n";
        os << "  lookahead (adaptive): " << default_lookahead << "\n";
        os << "  gapDelta (adaptive): " << default_gapDelta << "\n";
        
        os << "\nEpsilon search:\n";
        os << "  epsilon: " << default_epsilon << "\n";
        os << "  epsilonL2: " << default_epsilonL2 << "\n";
        os << "  epsilonIP: " << default_epsilonIP << "\n";
        os << "  min_candidates: " << min_candidates << "\n";
        os << "  max_candidates_cap: " << max_candidates_cap << "\n";
        
        os << "\nPerformance:\n";
        os << "  knn_lookahead_scale: " << knn_lookahead_scale << "\n";
        os << "  flush_threshold: " << flush_threshold << "\n";
        os << "  flush_offsets_each: " << (flush_offsets_each ? "yes" : "no") << "\n";
        os << "  parallel_merge: " << (parallel_merge ? "yes" : "no") << "\n";
        os << "  merge_threads: " << get_merge_threads() << "\n";
        
        os << "\nDebug: " << (debug ? "enabled" : "disabled") << "\n";
    }

    // Binary serialization
    void save(std::ostream& os) const {
        // Version marker
        uint32_t version = 1;
        os.write((char*)&version, sizeof(version));
        
        // Write all fields
        os.write((char*)this, sizeof(HnswConfig));
    }

    void load(std::istream& is) {
        // Read version
        uint32_t version;
        is.read((char*)&version, sizeof(version));
        
        if (version != 1) {
            throw std::runtime_error("Unsupported config version");
        }
        
        // Read all fields
        is.read((char*)this, sizeof(HnswConfig));
        
        if (!validate()) {
            throw std::runtime_error("Loaded invalid configuration");
        }
    }

    // String conversions
    static std::string metric_to_string(Metric m) {
        switch (m) {
            case Metric::L2: return "L2";
            case Metric::InnerProduct: return "InnerProduct";
            case Metric::Cosine: return "Cosine";
            default: return "Unknown";
        }
    }

    static std::string search_mode_to_string(SearchMode m) {
        switch (m) {
            case SearchMode::Knn: return "Knn";
            case SearchMode::Radius: return "Radius";
            case SearchMode::Relative: return "Relative";
            case SearchMode::Adaptive: return "Adaptive";
            case SearchMode::Epsilon: return "Epsilon";
            default: return "Unknown";
        }
    }

    static Metric string_to_metric(const std::string& s) {
        if (s == "L2" || s == "l2") return Metric::L2;
        if (s == "InnerProduct" || s == "IP" || s == "ip") return Metric::InnerProduct;
        if (s == "Cosine" || s == "cosine") return Metric::Cosine;
        throw std::runtime_error("Unknown metric: " + s);
    }

    static SearchMode string_to_search_mode(const std::string& s) {
        if (s == "knn" || s == "Knn") return SearchMode::Knn;
        if (s == "radius" || s == "Radius") return SearchMode::Radius;
        if (s == "relative" || s == "Relative") return SearchMode::Relative;
        if (s == "adaptive" || s == "Adaptive") return SearchMode::Adaptive;
        if (s == "epsilon" || s == "Epsilon") return SearchMode::Epsilon;
        throw std::runtime_error("Unknown search mode: " + s);
    }
};

// Builder pattern for easy construction
class HnswConfigBuilder {
private:
    HnswConfig cfg;

public:
    HnswConfigBuilder& with_max_elements(size_t n) { cfg.max_elements = n; return *this; }
    HnswConfigBuilder& with_M(size_t m) { cfg.M = m; return *this; }
    HnswConfigBuilder& with_ef_construction(size_t ef) { cfg.ef_construction = ef; return *this; }
    HnswConfigBuilder& with_ef_search(size_t ef) { cfg.ef_search = ef; return *this; }
    HnswConfigBuilder& with_metric(Metric m) { cfg.metric = m; return *this; }
    HnswConfigBuilder& with_search_mode(SearchMode m) { cfg.default_search_mode = m; return *this; }
    HnswConfigBuilder& with_debug(bool d) { cfg.debug = d; return *this; }
    HnswConfigBuilder& with_normalized_embeddings(bool n) { cfg.normalized_embeddings = n; return *this; }
    HnswConfigBuilder& with_default_k(size_t k) { cfg.default_k = k; return *this; }
    HnswConfigBuilder& with_epsilon(float e) { cfg.default_epsilon = e; return *this; }
    HnswConfigBuilder& with_flush_threshold(int t) { cfg.flush_threshold = t; return *this; }
    
    HnswConfig build() {
        if (!cfg.validate()) {
            throw std::runtime_error("Invalid configuration");
        }
        return cfg;
    }
};

// Example usage
int main() {
    std::cout << "=== Default Configuration ===\n";
    HnswConfig cfg1;
    cfg1.print();
    cfg1.validate();
    
    std::cout << "\n=== Custom Configuration (Builder Pattern) ===\n";
    HnswConfig cfg2 = HnswConfigBuilder()
        .with_max_elements(50000)
        .with_M(32)
        .with_ef_construction(400)
        .with_metric(Metric::L2)
        .with_search_mode(SearchMode::Epsilon)
        .with_epsilon(0.5f)
        .with_debug(false)
        .build();
    
    cfg2.print();
    
    std::cout << "\n=== Helper Methods ===\n";
    std::cout << "Effective epsilon: " << cfg2.get_epsilon() << "\n";
    std::cout << "Max candidates (auto): " << cfg2.get_max_candidates() << "\n";
    std::cout << "Merge threads: " << cfg2.get_merge_threads() << "\n";
    
    std::cout << "\n=== Serialization Test ===\n";
    // Save config
    std::ofstream ofs("config.bin", std::ios::binary);
    cfg2.save(ofs);
    ofs.close();
    
    // Load config
    HnswConfig cfg3;
    std::ifstream ifs("config.bin", std::ios::binary);
    cfg3.load(ifs);
    ifs.close();
    
    std::cout << "Loaded config:\n";
    cfg3.print();
    
    std::cout << "\n=== String Conversions ===\n";
    std::cout << "Metric: " << HnswConfig::metric_to_string(cfg2.metric) << "\n";
    std::cout << "Search mode: " << HnswConfig::search_mode_to_string(cfg2.default_search_mode) << "\n";
    
    Metric m = HnswConfig::string_to_metric("Cosine");
    std::cout << "Parsed 'Cosine' -> " << (int)m << "\n";
    
    return 0;
}

/*
 * FEATURES:
 * 
 * 1. Validation:
 *    - validate() checks all parameters for sanity
 *    - Warns about suboptimal settings
 *    - Called automatically in builder
 * 
 * 2. Helper methods:
 *    - get_epsilon(): Returns appropriate epsilon for current metric
 *    - get_max_candidates(): Applies cap if set
 *    - get_merge_threads(): Auto-detects if set to 0
 * 
 * 3. Serialization:
 *    - Binary save/load for persistence
 *    - Version marker for future compatibility
 *    - Validates on load
 * 
 * 4. Builder pattern:
 *    - Fluent interface for construction
 *    - Validates at build() time
 *    - Easier to construct complex configs
 * 
 * 5. String conversions:
 *    - Parse from config files
 *    - Display in logs
 *    - Bidirectional conversion
 * 
 * 6. Print method:
 *    - Human-readable output
 *    - Great for debugging
 *    - Can write to any ostream
 * 
 * USAGE EXAMPLES:
 * 
 * // Simple construction
 * HnswConfig cfg;
 * cfg.max_elements = 50000;
 * cfg.metric = Metric::L2;
 * 
 * // Builder pattern
 * auto cfg = HnswConfigBuilder()
 *     .with_max_elements(50000)
 *     .with_metric(Metric::L2)
 *     .build();
 * 
 * // Get epsilon for current metric
 * float eps = cfg.get_epsilon();
 * 
 * // Save/load
 * cfg.save(ofs);
 * cfg.load(ifs);
 * 
 * // Validate
 * if (!cfg.validate()) {
 *     // Handle error
 * }
 */


/*

// Simple
HnswConfig cfg;
cfg.max_elements = 50000;

// Builder
auto cfg = HnswConfigBuilder()
    .with_max_elements(50000)
    .with_metric(Metric::L2)
    .with_epsilon(0.5f)
    .build();

// Helpers
float eps = cfg.get_epsilon();
cfg.print();

*/
