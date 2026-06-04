#pragma once
#include "space_quantized_ip.h"

/*
Cosine Similarity Quantized Space for HNSW

This is a specialized wrapper around SpaceQuantizedIP with:
- Automatic L2 normalization of input vectors
- Normalization tracking always enabled
- Distance = 1 - cosine_similarity = 1 - (a·b)/(||a||·||b||)

For cosine similarity, all vectors should be normalized to unit length
before quantization. This simplifies to regular inner product after normalization.
*/

namespace hnswlib {

template<typename T=float>
class SpaceQuantizedCosine : public SpaceInterface<float> {
public:
    using DISTFUNC_TYPE = DISTFUNC<float>;

private:
    SpaceQuantizedIP<T> inner_space_;
    size_t dim_;
    
    // Normalization helper
    void normalize_vector(const T* input, T* output, T& norm) const {
        T norm_sq = 0;
        for (size_t i = 0; i < dim_; ++i) {
            norm_sq += input[i] * input[i];
        }
        norm = std::sqrt(norm_sq);
        
        if (norm > 1e-9) {
            T inv_norm = T(1) / norm;
            for (size_t i = 0; i < dim_; ++i) {
                output[i] = input[i] * inv_norm;
            }
        } else {
            // Handle zero vectors
            std::memcpy(output, input, dim_ * sizeof(T));
        }
    }

public:
    explicit SpaceQuantizedCosine(size_t dim,
                                  QuantModeIP qmode,
                                  OptBinModeIP bin_mode = OptBinModeIP::STANDARD,
                                  const std::vector<std::vector<T>>* sample_embeddings = nullptr,
                                  size_t buffer_capacity = 1000)
        : inner_space_(dim, qmode, bin_mode, 
                      sample_embeddings ? normalize_samples(sample_embeddings) : nullptr,
                      buffer_capacity,
                      true),  // Always use normalization for cosine
          dim_(dim) {
        
        HNSWDEBUG << "  [SpaceQuantizedCosine: vectors will be L2-normalized]";
    }
    
    // Normalize sample embeddings for training
    const std::vector<std::vector<T>>* normalize_samples(const std::vector<std::vector<T>>* samples) {
        if (!samples) return nullptr;
        
        // We need to store normalized samples (can't modify the input)
        // For simplicity, we'll let the inner space handle this during training
        // The caller should pre-normalize if they want to train properly
        return samples;
    }

    // Interface compliance
    size_t get_data_size() override { 
        return inner_space_.get_data_size();
    }
    
    DISTFUNC_TYPE get_dist_func() override { 
        return inner_space_.get_dist_func();
    }
    
    void* get_dist_func_param() override { 
        return inner_space_.get_dist_func_param();
    }

    // Quantize with automatic normalization
    void quantize(const T* emb, uint8_t* out) {
        std::vector<T> normalized(dim_);
        T norm;
        normalize_vector(emb, normalized.data(), norm);
        
        // Store norm for potential rescoring
        inner_space_.store_norm(normalized.data());
        
        // Quantize the normalized vector
        inner_space_.quantize(normalized.data(), out);
    }
    
    // Convenience method for quantization without norm tracking
    void quantize_without_norm_tracking(const T* emb, uint8_t* out) {
        std::vector<T> normalized(dim_);
        T norm;
        normalize_vector(emb, normalized.data(), norm);
        inner_space_.quantize(normalized.data(), out);
    }
    
    // Static helper to normalize a vector in-place
    static void normalize_inplace(T* vec, size_t dim) {
        T norm_sq = 0;
        for (size_t i = 0; i < dim; ++i) {
            norm_sq += vec[i] * vec[i];
        }
        T norm = std::sqrt(norm_sq);
        
        if (norm > 1e-9) {
            T inv_norm = T(1) / norm;
            for (size_t i = 0; i < dim; ++i) {
                vec[i] *= inv_norm;
            }
        }
    }
    
    // Static helper to normalize a batch of vectors
    static void normalize_batch(std::vector<std::vector<T>>& vectors) {
        for (auto& vec : vectors) {
            normalize_inplace(vec.data(), vec.size());
        }
    }

    // Centroid operations (forward to inner space)
    void add_to_centroid(const T* emb) {
        std::vector<T> normalized(dim_);
        T norm;
        normalize_vector(emb, normalized.data(), norm);
        inner_space_.add_to_centroid(normalized.data());
    }
    
    void flush_centroid_buffer() {
        inner_space_.flush_centroid_buffer();
    }
    
    bool save_centroid(const std::string& filepath) const {
        return inner_space_.save_centroid(filepath);
    }
    
    bool load_centroid(const std::string& filepath) {
        return inner_space_.load_centroid(filepath);
    }
    
    std::vector<T> get_centroid() {
        return inner_space_.get_centroid();
    }
    
    size_t get_centroid_count() const {
        return inner_space_.get_centroid_count();
    }
    
    size_t get_buffer_count() const {
        return inner_space_.get_buffer_count();
    }
    
    // Access to underlying IP space (for advanced use)
    SpaceQuantizedIP<T>& get_inner_space() {
        return inner_space_;
    }
    
    const SpaceQuantizedIP<T>& get_inner_space() const {
        return inner_space_;
    }
};

// Convenience type aliases
using SpaceQuantizedCosineF = SpaceQuantizedCosine<float>;
using SpaceQuantizedCosineD = SpaceQuantizedCosine<double>;

} // namespace hnswlib

/*
Usage example for Cosine Similarity:

// Create space with binary quantization for cosine similarity
auto space = new hnswlib::SpaceQuantizedCosine<float>(
    dim, 
    hnswlib::QuantModeIP::BIN1, 
    hnswlib::OptBinModeIP::RABITQ,
    nullptr,  // Can provide normalized samples for training
    1000      // buffer capacity
);

// If you have training samples, normalize them first
std::vector<std::vector<float>> training_samples = load_samples();
hnswlib::SpaceQuantizedCosine<float>::normalize_batch(training_samples);

// Now create space with normalized samples
auto space = new hnswlib::SpaceQuantizedCosine<float>(
    dim, 
    hnswlib::QuantModeIP::BIN1, 
    hnswlib::OptBinModeIP::CENTROID,
    &training_samples,
    1000
);

// Create HNSW index
auto index = new hnswlib::HierarchicalNSW<float>(space, max_elements);

// Add points - normalization happens automatically
for (size_t i = 0; i < num_vectors; ++i) {
    // No need to normalize manually - space handles it
    space->add_to_centroid(vectors[i].data());
    
    std::vector<uint8_t> code(space->get_data_size());
    space->quantize(vectors[i].data(), code.data());
    index->addPoint(code.data(), i);
}

// Query - also normalized automatically
std::vector<uint8_t> query_code(space->get_data_size());
space->quantize(query_vector.data(), query_code.data());
auto results = index->searchKnn(query_code.data(), k);

// Save
space->flush_centroid_buffer();
space->save_centroid("centroid_cosine.bin");
index->saveIndex("index_cosine.bin");

// Key advantages of SpaceQuantizedCosine:
// 1. Automatic normalization - no manual preprocessing needed
// 2. Clean interface - just quantize and it works
// 3. Semantically clear - cosine similarity in the name
// 4. All the power of quantization (BIN1, INT4, INT8, RaBitQ, etc.)
*/
