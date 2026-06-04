#include "hnswlib/hnswlib.h"
#include <fstream>
#include <iostream>
#include <memory>

// Method 1: Peek at the index file to get element count
// Actual HNSWlib format: [unknown/offset, max_elements, cur_element_count, ...]
std::pair<size_t, size_t> peek_index_elements(std::istream& ifs) {
    // Save current position
    std::streampos original_pos = ifs.tellg();
    
    // Read first three size_t values
    size_t val1, val2, val3;
    ifs.read((char*)&val1, sizeof(size_t));
    ifs.read((char*)&val2, sizeof(size_t));
    ifs.read((char*)&val3, sizeof(size_t));
    
    // Restore position
    ifs.seekg(original_pos);
    
    size_t max_elements = val2;   // Second value
    size_t cur_elements = val3;   // Third value
    
    std::cout << "Peeked HNSWlib header: val1=" << val1 
              << ", max=" << val2 
              << ", cur=" << val3 << "\n";
    
    return {cur_elements, max_elements};
}

// Method 2: Better approach - use the alternate constructor
std::unique_ptr<hnswlib::HierarchicalNSW<float>> load_index_proper(
    const std::string& filename,
    hnswlib::SpaceInterface<float>* space,
    size_t fallback_max_elements = 100000)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open index file");
    }
    
    // Peek at the values
    auto [cur_count, max_from_file] = peek_index_elements(ifs);
    
    std::cout << "Index file has cur=" << cur_count 
              << ", max=" << max_from_file << " elements\n";
    
    // Use the max from file (it already includes the saved max_elements)
    size_t max_elements = std::max(max_from_file, fallback_max_elements);
    
    std::cout << "Allocating space for " << max_elements << " elements\n";
    
    // Create index with appropriate size
    auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        space, max_elements);
    
    // Load the index
    index->loadIndex(ifs, space);
    
    std::cout << "Loaded " << index->cur_element_count << " elements\n";
    
    return index;
}

// Method 3: Even better - use HNSWlib's built-in constructor
std::unique_ptr<hnswlib::HierarchicalNSW<float>> load_index_best(
    const std::string& filename,
    hnswlib::SpaceInterface<float>* space)
{
    // This constructor reads max_elements from the file automatically!
    auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        space, filename);
    
    std::cout << "Loaded index with " << index->cur_element_count 
              << " / " << index->max_elements_ << " elements\n";
    
    return index;
}

// Method 4: Load with automatic resizing if needed
std::unique_ptr<hnswlib::HierarchicalNSW<float>> load_index_with_resize(
    std::istream& ifs,
    hnswlib::SpaceInterface<float>* space,
    size_t initial_max_elements = 100000)
{
    // Peek at values
    auto [cur_count, max_from_file] = peek_index_elements(ifs);
    
    // Use the max_elements that was saved in the file
    size_t max_elements;
    if (max_from_file > initial_max_elements) {
        std::cout << "INFO: Index was saved with max_elements=" << max_from_file 
                  << ", cfg has " << initial_max_elements << "\n";
        std::cout << "Using " << max_from_file << " to match saved index\n";
        max_elements = max_from_file;
    } else {
        max_elements = initial_max_elements;
    }
    
    auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        space, max_elements);
    
    index->loadIndex(ifs, space);
    
    return index;
}

// Method 5: Your current code - FIXED
class BertIndex {
private:
    struct IndexMetadata {
        uint32_t version;
        uint32_t metric;
        bool normalized;
        size_t dimension;
        size_t element_count;  // ADD THIS!
    };
    
    std::unique_ptr<hnswlib::SpaceInterface<float>> space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
    
    struct Config {
        size_t max_elements;
        // ... other config
    } cfg;
    
public:
    void save_index(const std::string& filename) {
        std::ofstream ofs(filename, std::ios::binary);
        
        // Save metadata INCLUDING element count
        IndexMetadata meta;
        meta.version = 1;
        meta.metric = 2;  // your metric
        meta.normalized = true;
        meta.dimension = 384;
        meta.element_count = index->cur_element_count;  // SAVE THIS!
        
        ofs.write((char*)&meta, sizeof(meta));
        
        // Save index
        index->saveIndex(ofs);
    }
    
    void load_index(const std::string& filename) {
        std::ifstream ifs(filename, std::ios::binary);
        
        // Read metadata
        IndexMetadata meta;
        ifs.read((char*)&meta, sizeof(meta));
        
        std::cout << "Index metadata: " << meta.element_count << " elements\n";
        
        // Also peek at HNSWlib's header to get its max_elements
        auto [cur_from_file, max_from_file] = peek_index_elements(ifs);
        
        std::cout << "HNSWlib header: cur=" << cur_from_file 
                  << ", max=" << max_from_file << "\n";
        
        // Determine appropriate max_elements
        // Use the maximum of all three values
        size_t max_elements = std::max({
            meta.element_count,
            max_from_file,
            cfg.max_elements
        });
        
        std::cout << "Allocating index for " << max_elements << " elements\n";
        
        // Create space
        space = AllocateSpace((Metric)meta.metric, meta.dimension);
        
        // Create index with CORRECT size
        index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space.get(), max_elements);
        
        // Load index data
        index->loadIndex(ifs, space.get());
        
        std::cout << "Loaded " << index->cur_element_count << " elements\n";
    }
};

// Example usage
int main() {
    try {
        std::cout << "=== Method 1: Peek at file ===\n";
        std::ifstream ifs1("test.hix", std::ios::binary);
        auto [cur, max] = peek_index_elements(ifs1);
        std::cout << "File has cur=" << cur << ", max=" << max << " elements\n\n";
        
        std::cout << "=== Method 2: Load with proper sizing ===\n";
        hnswlib::L2Space space(128);
        auto index2 = load_index_proper("test.hix", &space, 1000);
        std::cout << "\n";
        
        std::cout << "=== Method 3: Use filename constructor (BEST) ===\n";
        // This is the cleanest approach - HNSWlib handles it!
        auto index3 = load_index_best("test.hix", &space);
        std::cout << "\n";
        
        std::cout << "=== Method 4: Load with auto-resize ===\n";
        std::ifstream ifs4("test.hix", std::ios::binary);
        auto index4 = load_index_with_resize(ifs4, &space, 1000);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
    
    return 0;
}

/*
 * RECOMMENDATIONS:
 * 
 * 1. BEST APPROACH - Use filename constructor:
 *    auto index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
 *        space.get(), filename);
 *    
 *    This reads max_elements from the file automatically!
 * 
 * 2. If using stream-based loading:
 *    a) Store element_count in YOUR metadata file
 *    b) Read it before creating HierarchicalNSW
 *    c) Use max(element_count * 2, cfg.max_elements)
 * 
 * 3. Why 2x element_count?
 *    - Leaves room for growth
 *    - Avoid immediate reallocation
 *    - Balance between memory and flexibility
 * 
 * 4. Alternative: Peek at index header
 *    - Read first few bytes to get max_elements
 *    - Then seekg back and load
 *    - More complex but works
 * 
 * 5. Your current code issue:
 *    cfg.max_elements might be smaller than index size!
 *    
 *    Solution: Save element_count in your metadata,
 *             read it first, then allocate properly
 * 
 * 6. Memory considerations:
 *    max_elements affects memory usage
 *    Set to: max(actual_count * 1.5, desired_max)
 * 
 * 7. Error handling:
 *    if (element_count > max_elements) {
 *        throw or auto-resize
 *    }
 */
