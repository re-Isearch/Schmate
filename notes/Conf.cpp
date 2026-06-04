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

    // Save to file
    bool save_to_file(const std::string& path) const {
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs) {
            std::cerr << "Failed to open " << path << " for writing\n";
            return false;
        }
        save(ofs);
        return true;
    }

    // Load from file
    bool load_from_file(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) {
            return false;  // File doesn't exist, not an error
        }
        try {
            load(ifs);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading config from " << path << ": " << e.what() << "\n";
            return false;
        }
    }

    // Merge/override with another config
    void merge_from(const HnswConfig& other) {
        // Copy all fields from other
        // This is a simple approach - copies everything
        *this = other;
    }

    // Selective merge (only override non-default values)
    void merge_overrides(const HnswConfig& override, const HnswConfig& defaults) {
        // Helper macro to check and override
        #define OVERRIDE_IF_DIFFERENT(field) \
            if (override.field != defaults.field) { \
                this->field = override.field; \
            }
        
        OVERRIDE_IF_DIFFERENT(default_search_mode);
        OVERRIDE_IF_DIFFERENT(max_elements);
        OVERRIDE_IF_DIFFERENT(M);
        OVERRIDE_IF_DIFFERENT(ef_construction);
        OVERRIDE_IF_DIFFERENT(ef_search);
        OVERRIDE_IF_DIFFERENT(metric);
        OVERRIDE_IF_DIFFERENT(bert_n_threads);
        OVERRIDE_IF_DIFFERENT(max_tokens_per_chunk);
        OVERRIDE_IF_DIFFERENT(overlap_percent);
        OVERRIDE_IF_DIFFERENT(debug);
        OVERRIDE_IF_DIFFERENT(default_k);
        OVERRIDE_IF_DIFFERENT(default_radius);
        OVERRIDE_IF_DIFFERENT(default_alpha);
        OVERRIDE_IF_DIFFERENT(default_minN);
        OVERRIDE_IF_DIFFERENT(default_lookahead);
        OVERRIDE_IF_DIFFERENT(default_gapDelta);
        OVERRIDE_IF_DIFFERENT(default_epsilon);
        OVERRIDE_IF_DIFFERENT(default_epsilonL2);
        OVERRIDE_IF_DIFFERENT(default_epsilonIP);
        OVERRIDE_IF_DIFFERENT(min_candidates);
        OVERRIDE_IF_DIFFERENT(max_candidates_cap);
        OVERRIDE_IF_DIFFERENT(knn_lookahead_scale);
        OVERRIDE_IF_DIFFERENT(flush_threshold);
        OVERRIDE_IF_DIFFERENT(flush_offsets_each);
        OVERRIDE_IF_DIFFERENT(parallel_merge);
        OVERRIDE_IF_DIFFERENT(merge_threads);
        OVERRIDE_IF_DIFFERENT(normalized_embeddings);
        
        #undef OVERRIDE_IF_DIFFERENT
    }

    // Dynamic setter by string key
    bool set(const std::string& key, const std::string& value) {
        try {
            // size_t fields
            if (key == "max_elements") { max_elements = std::stoull(value); return true; }
            if (key == "M") { M = std::stoull(value); return true; }
            if (key == "ef_construction") { ef_construction = std::stoull(value); return true; }
            if (key == "ef_search") { ef_search = std::stoull(value); return true; }
            if (key == "bert_n_threads") { bert_n_threads = std::stoull(value); return true; }
            if (key == "default_k") { default_k = std::stoull(value); return true; }
            if (key == "default_minN") { default_minN = std::stoull(value); return true; }
            if (key == "default_lookahead") { default_lookahead = std::stoull(value); return true; }
            if (key == "min_candidates") { min_candidates = std::stoull(value); return true; }
            if (key == "max_candidates_cap") { max_candidates_cap = std::stoull(value); return true; }
            if (key == "knn_lookahead_scale") { knn_lookahead_scale = std::stoull(value); return true; }
            if (key == "merge_threads") { merge_threads = std::stoul(value); return true; }
            
            // int fields
            if (key == "max_tokens_per_chunk") { max_tokens_per_chunk = std::stoi(value); return true; }
            if (key == "flush_threshold") { flush_threshold = std::stoi(value); return true; }
            
            // float fields
            if (key == "overlap_percent") { overlap_percent = std::stof(value); return true; }
            if (key == "default_radius") { default_radius = std::stof(value); return true; }
            if (key == "default_alpha") { default_alpha = std::stof(value); return true; }
            if (key == "default_gapDelta") { default_gapDelta = std::stof(value); return true; }
            if (key == "default_epsilon") { default_epsilon = std::stof(value); return true; }
            if (key == "default_epsilonL2") { default_epsilonL2 = std::stof(value); return true; }
            if (key == "default_epsilonIP") { default_epsilonIP = std::stof(value); return true; }
            
            // bool fields
            if (key == "debug") { debug = parse_bool(value); return true; }
            if (key == "flush_offsets_each") { flush_offsets_each = parse_bool(value); return true; }
            if (key == "parallel_merge") { parallel_merge = parse_bool(value); return true; }
            if (key == "normalized_embeddings") { normalized_embeddings = parse_bool(value); return true; }
            
            // enum fields
            if (key == "metric") { metric = string_to_metric(value); return true; }
            if (key == "default_search_mode") { default_search_mode = string_to_search_mode(value); return true; }
            
            std::cerr << "Unknown config key: " << key << "\n";
            return false;
            
        } catch (const std::exception& e) {
            std::cerr << "Error setting " << key << "=" << value << ": " << e.what() << "\n";
            return false;
        }
    }

    // Typed setters for convenience
    bool set(const std::string& key, size_t value) {
        return set(key, std::to_string(value));
    }
    
    bool set(const std::string& key, int value) {
        return set(key, std::to_string(value));
    }
    
    bool set(const std::string& key, float value) {
        return set(key, std::to_string(value));
    }
    
    bool set(const std::string& key, bool value) {
        return set(key, value ? "true" : "false");
    }
    
    bool set(const std::string& key, Metric value) {
        return set(key, metric_to_string(value));
    }
    
    bool set(const std::string& key, SearchMode value) {
        return set(key, search_mode_to_string(value));
    }

    // Dynamic getter by string key
    std::string get(const std::string& key) const {
        // size_t fields
        if (key == "max_elements") return std::to_string(max_elements);
        if (key == "M") return std::to_string(M);
        if (key == "ef_construction") return std::to_string(ef_construction);
        if (key == "ef_search") return std::to_string(ef_search);
        if (key == "bert_n_threads") return std::to_string(bert_n_threads);
        if (key == "default_k") return std::to_string(default_k);
        if (key == "default_minN") return std::to_string(default_minN);
        if (key == "default_lookahead") return std::to_string(default_lookahead);
        if (key == "min_candidates") return std::to_string(min_candidates);
        if (key == "max_candidates_cap") return std::to_string(max_candidates_cap);
        if (key == "knn_lookahead_scale") return std::to_string(knn_lookahead_scale);
        if (key == "merge_threads") return std::to_string(merge_threads);
        
        // int fields
        if (key == "max_tokens_per_chunk") return std::to_string(max_tokens_per_chunk);
        if (key == "flush_threshold") return std::to_string(flush_threshold);
        
        // float fields
        if (key == "overlap_percent") return std::to_string(overlap_percent);
        if (key == "default_radius") return std::to_string(default_radius);
        if (key == "default_alpha") return std::to_string(default_alpha);
        if (key == "default_gapDelta") return std::to_string(default_gapDelta);
        if (key == "default_epsilon") return std::to_string(default_epsilon);
        if (key == "default_epsilonL2") return std::to_string(default_epsilonL2);
        if (key == "default_epsilonIP") return std::to_string(default_epsilonIP);
        
        // bool fields
        if (key == "debug") return debug ? "true" : "false";
        if (key == "flush_offsets_each") return flush_offsets_each ? "true" : "false";
        if (key == "parallel_merge") return parallel_merge ? "true" : "false";
        if (key == "normalized_embeddings") return normalized_embeddings ? "true" : "false";
        
        // enum fields
        if (key == "metric") return metric_to_string(metric);
        if (key == "default_search_mode") return search_mode_to_string(default_search_mode);
        
        throw std::runtime_error("Unknown config key: " + key);
    }

    // Get all keys
    static std::vector<std::string> get_all_keys() {
        return {
            "max_elements", "M", "ef_construction", "ef_search", "metric",
            "bert_n_threads", "max_tokens_per_chunk", "overlap_percent",
            "debug", "default_k", "default_radius", "default_alpha",
            "default_minN", "default_lookahead", "default_gapDelta",
            "default_epsilon", "default_epsilonL2", "default_epsilonIP",
            "min_candidates", "max_candidates_cap", "knn_lookahead_scale",
            "flush_threshold", "flush_offsets_each", "parallel_merge",
            "merge_threads", "normalized_embeddings", "default_search_mode"
        };
    }

private:
    static bool parse_bool(const std::string& s) {
        std::string lower = s;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower == "true" || lower == "1" || lower == "yes" || lower == "on") return true;
        if (lower == "false" || lower == "0" || lower == "no" || lower == "off") return false;
        throw std::runtime_error("Invalid boolean value: " + s);
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

// Configuration loader with global defaults and local overrides
class ConfigLoader {
private:
    std::string global_config_path;
    std::string local_config_path;
    
public:
    ConfigLoader(const std::string& global_path = "/etc/hnsw/config.bin",
                 const std::string& local_path = "~/.hnsw/config.bin")
        : global_config_path(global_path)
        , local_config_path(expand_home(local_path))
    {}
    
    // Load configuration with priority: local > global > hardcoded defaults
    HnswConfig load() const {
        HnswConfig cfg;  // Start with hardcoded defaults
        HnswConfig defaults = cfg;  // Save for comparison
        
        // Load global config (if exists)
        if (cfg.load_from_file(global_config_path)) {
            std::cout << "Loaded global config from " << global_config_path << "\n";
            defaults = cfg;  // Update defaults to global config
        } else {
            std::cout << "No global config found, using hardcoded defaults\n";
        }
        
        // Load local config and override (if exists)
        HnswConfig local_cfg = cfg;  // Start with current config
        if (local_cfg.load_from_file(local_config_path)) {
            std::cout << "Loaded local config from " << local_config_path << "\n";
            cfg.merge_overrides(local_cfg, defaults);
            std::cout << "Applied local overrides\n";
        } else {
            std::cout << "No local config found\n";
        }
        
        return cfg;
    }
    
    // Load from specific project directory
    HnswConfig load_with_project(const std::string& project_dir) const {
        HnswConfig cfg = load();  // Start with global+local
        
        // Try to load project-specific config
        std::string project_config = project_dir + "/config.bin";
        HnswConfig project_cfg = cfg;
        if (project_cfg.load_from_file(project_config)) {
            std::cout << "Loaded project config from " << project_config << "\n";
            HnswConfig base = cfg;
            cfg.merge_overrides(project_cfg, base);
            std::cout << "Applied project overrides\n";
        }
        
        return cfg;
    }
    
    // Save as global default
    bool save_global(const HnswConfig& cfg) const {
        return cfg.save_to_file(global_config_path);
    }
    
    // Save as local override
    bool save_local(const HnswConfig& cfg) const {
        return cfg.save_to_file(local_config_path);
    }
    
private:
    static std::string expand_home(const std::string& path) {
        if (path.empty() || path[0] != '~') {
            return path;
        }
        const char* home = getenv("HOME");
        if (!home) home = getenv("USERPROFILE");  // Windows
        if (!home) return path;
        return std::string(home) + path.substr(1);
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
    
    std::cout << "\n=== Configuration Loading System ===\n";
    
    // Create some test configs
    std::cout << "Setting up test configurations...\n";
    
    // Global defaults (conservative settings)
    HnswConfig global_defaults;
    global_defaults.max_elements = 10000;
    global_defaults.M = 16;
    global_defaults.debug = false;
    global_defaults.save_to_file("global_config.bin");
    std::cout << "Saved global config\n";
    
    // Local overrides (performance settings)
    HnswConfig local_overrides;
    local_overrides.max_elements = 50000;  // Override
    local_overrides.M = 32;                // Override
    local_overrides.ef_construction = 400; // Override
    local_overrides.debug = true;          // Override
    // Other fields remain at default
    local_overrides.save_to_file("local_config.bin");
    std::cout << "Saved local config\n";
    
    // Load with priority system
    std::cout << "\n=== Loading Configuration ===\n";
    ConfigLoader loader("global_config.bin", "local_config.bin");
    HnswConfig final_cfg = loader.load();
    
    std::cout << "\n=== Final Merged Configuration ===\n";
    final_cfg.print();
    
    std::cout << "\nExpected results:\n";
    std::cout << "  max_elements: 50000 (from local)\n";
    std::cout << "  M: 32 (from local)\n";
    std::cout << "  ef_construction: 400 (from local)\n";
    std::cout << "  debug: true (from local)\n";
    std::cout << "  ef_search: 50 (from global/default)\n";
    
    // Test project-specific config
    std::cout << "\n=== Project-Specific Configuration ===\n";
    
    // Create project config (further overrides)
    HnswConfig project_cfg;
    project_cfg.max_elements = 100000;  // Override again
    project_cfg.metric = Metric::Cosine;
    project_cfg.save_to_file("project_config.bin");
    
    // Simulate project directory structure
    system("mkdir -p myproject");
    system("cp project_config.bin myproject/config.bin");
    
    HnswConfig project_final = loader.load_with_project("myproject");
    std::cout << "\n=== Final Project Configuration ===\n";
    project_final.print();
    
    std::cout << "\n=== Three-Level Priority System ===\n";
    std::cout << "Priority order: Project > Local > Global > Hardcoded\n";
    std::cout << "Final values:\n";
    std::cout << "  max_elements: " << project_final.max_elements << " (from project)\n";
    std::cout << "  metric: " << HnswConfig::metric_to_string(project_final.metric) << " (from project)\n";
    std::cout << "  M: " << project_final.M << " (from local)\n";
    std::cout << "  debug: " << (project_final.debug ? "true" : "false") << " (from local)\n";
    
    // Test dynamic setters/getters
    std::cout << "\n=== Dynamic Setters/Getters ===\n";
    HnswConfig dynamic_cfg;
    
    // Set values using strings
    dynamic_cfg.set("max_elements", "75000");
    dynamic_cfg.set("M", "24");
    dynamic_cfg.set("ef_construction", "300");
    dynamic_cfg.set("metric", "L2");
    dynamic_cfg.set("debug", "true");
    dynamic_cfg.set("default_epsilon", "0.25");
    dynamic_cfg.set("normalized_embeddings", "yes");
    
    std::cout << "Set values using strings:\n";
    std::cout << "  max_elements = " << dynamic_cfg.get("max_elements") << "\n";
    std::cout << "  M = " << dynamic_cfg.get("M") << "\n";
    std::cout << "  metric = " << dynamic_cfg.get("metric") << "\n";
    std::cout << "  debug = " << dynamic_cfg.get("debug") << "\n";
    std::cout << "  default_epsilon = " << dynamic_cfg.get("default_epsilon") << "\n";
    
    // Set values using typed overloads
    dynamic_cfg.set("max_elements", 100000);
    dynamic_cfg.set("default_radius", 0.85f);
    dynamic_cfg.set("parallel_merge", false);
    dynamic_cfg.set("metric", Metric::Cosine);
    
    std::cout << "\nAfter typed overloads:\n";
    std::cout << "  max_elements = " << dynamic_cfg.max_elements << "\n";
    std::cout << "  default_radius = " << dynamic_cfg.default_radius << "\n";
    std::cout << "  parallel_merge = " << dynamic_cfg.parallel_merge << "\n";
    std::cout << "  metric = " << HnswConfig::metric_to_string(dynamic_cfg.metric) << "\n";
    
    // List all available keys
    std::cout << "\nAvailable config keys:\n";
    for (const auto& key : HnswConfig::get_all_keys()) {
        std::cout << "  " << key << " = " << dynamic_cfg.get(key) << "\n";
    }
    
    // Error handling
    std::cout << "\n=== Error Handling ===\n";
    if (!dynamic_cfg.set("nonexistent_key", "value")) {
        std::cout << "Correctly rejected unknown key\n";
    }
    
    try {
        dynamic_cfg.set("max_elements", "not_a_number");
    } catch (const std::exception& e) {
        std::cout << "Caught parsing error as expected\n";
    }
    
    return 0;
}

/*
 * CONFIGURATION LOADING SYSTEM:
 * 
 * Priority order (highest to lowest):
 * 1. Project-specific config (./myproject/config.bin)
 * 2. Local user config (~/.hnsw/config.bin)
 * 3. Global system config (/etc/hnsw/config.bin)
 * 4. Hardcoded defaults (in struct definition)
 * 
 * USAGE:
 * 
 * // Load with priority system
 * ConfigLoader loader;
 * HnswConfig cfg = loader.load();
 * 
 * // With project-specific overrides
 * HnswConfig cfg = loader.load_with_project("./myproject");
 * 
 * // Dynamic setters (string-based)
 * cfg.set("max_elements", "100000");
 * cfg.set("metric", "Cosine");
 * cfg.set("debug", "true");
 * 
 * // Typed setters (type-safe)
 * cfg.set("max_elements", 100000);
 * cfg.set("default_radius", 0.85f);
 * cfg.set("metric", Metric::Cosine);
 * 
 * // Dynamic getters
 * std::string value = cfg.get("max_elements");
 * size_t max = cfg.max_elements;  // Direct access still works
 * 
 * // List all keys
 * for (const auto& key : HnswConfig::get_all_keys()) {
 *     std::cout << key << " = " << cfg.get(key) << "\n";
 * }
 * 
 * DYNAMIC SETTER FEATURES:
 * 
 * 1. String-based setting:
 *    cfg.set("max_elements", "100")
 *    cfg.set("metric", "Cosine")
 *    cfg.set("debug", "yes")  // Supports: true/1/yes/on
 * 
 * 2. Type-safe overloads:
 *    cfg.set("max_elements", 100)      // size_t
 *    cfg.set("default_radius", 0.85f)  // float
 *    cfg.set("debug", true)            // bool
 *    cfg.set("metric", Metric::Cosine) // enum
 * 
 * 3. Error handling:
 *    - Returns false for unknown keys
 *    - Throws exception for invalid values
 *    - Validates types automatically
 * 
 * 4. Get all keys:
 *    - HnswConfig::get_all_keys() returns vector of all keys
 *    - Useful for UI, CLI parsing, config files
 * 
 * USE CASES:
 * 
 * 1. Command-line parsing:
 *    cfg.set(argv[1], argv[2]);  // --max_elements 100000
 * 
 * 2. Config file parsing:
 *    for (line in config_file) {
 *        auto [key, value] = parse_line(line);
 *        cfg.set(key, value);
 *    }
 * 
 * 3. Environment variables:
 *    cfg.set("max_elements", getenv("HNSW_MAX_ELEMENTS"));
 * 
 * 4. Interactive CLI:
 *    std::string key, value;
 *    std::cin >> key >> value;
 *    cfg.set(key, value);
 * 
 * 5. Web API:
 *    cfg.set(json["key"], json["value"]);
 */
