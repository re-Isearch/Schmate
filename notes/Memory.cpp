#include "hnswlib/hnswlib.h"
#include <fstream>
#include <iostream>
#include <memory>

// Method 1: Peek at the index file to get element count
size_t peek_index_element_count(std::istream& ifs) {
    // Save current position
    std::streampos original_pos = ifs.tellg();
    
    // HNSWlib saves these values at the start of the file (in order):
    size_t max_elements;
    size_t cur_element_count;
    size_t size_data_per_element;
    size_t label_offset;
    size_t offsetLevel0;
    size_t max_level;
    size_t enterpoint_node;
    size_t maxM;
    size_t maxM0;
    size_t M;
    size_t mult;
    size_t ef_construction;
    
    // Read the header
    ifs.read((char*)&max_elements, sizeof(size_t));
    ifs.read((char*)&cur_element_count, sizeof(size_t));
    
    // Restore position
    ifs.seekg(original_pos);
    
    return cur_element_count;
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
    
    // Peek at the actual element count
    size_t actual_count = peek_index_element_count(ifs);
    
    std::cout << "Index contains " << actual_count << " elements\n";
    
    // Use the larger of actual count or fallback, with some headroom
    size_t max_elements = std::max(actual_count * 2, fallback_max_elements);
    
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
    // Peek at actual count
    size_t actual_count = peek_index_element_count(ifs);
    
    // Determine appropriate max_elements
    size_t max_elements;
    if (actual_count > initial_max_elements) {
        std::cout << "WARNING: Index has " << actual_count 
                  << " elements, but cfg.max_elements=" << initial_max_elements << "\n";
        std::cout << "Increasing to " << (actual_count * 2) << "\n";
        max_elements = actual_count * 2;  // 2x for future growth
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
        
        // Determine appropriate max_elements
        size_t max_elements;
        if (meta.element_count > cfg.max_elements) {
            std::cout << "WARNING: Index has more elements than configured max\n";
            std::cout << "  Index: " << meta.element_count 
                      << ", Config: " << cfg.max_elements << "\n";
            
            // Use the larger value with headroom
            max_elements = meta.element_count * 2;
        } else {
            max_elements = cfg.max_elements;
        }
        
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
        size_t count = peek_index_element_count(ifs1);
        std::cout << "File contains " << count << " elements\n\n";
        
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
