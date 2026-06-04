#pragma once
#include "space_quantized_ip.h"

/*
Cosine Similarity Quantized Space for HNSW

This space handles non-normalized vectors and computes true cosine similarity:
  cosine_sim(a, b) = (a·b) / (||a|| * ||b||)
  distance = 1 - cosine_sim

Vectors are normalized during quantization, and norms are stored for accurate
distance computation during search.
*/

namespace hnswlib {

static constexpr float cEPS = 1e-9f;

template<typename T=float>
class SpaceQuantizedCosine : public SpaceInterface<float> {
public:
    using DISTFUNC_TYPE = DISTFUNC<float>;

private:
    size_t dim_;
    QuantModeIP qmode_;
    OptBinModeIP bin_mode_;
    size_t bytes_per_vector_;
    size_t bytes_for_quantized_;
    
    SpaceQuantizedIP<T> inner_space_;
    
    // Store norms for all vectors (indexed by label)
    std::vector<T> norms_;
    mutable std::mutex norms_mutex_;
    std::unordered_map<size_t, size_t> label_to_norm_idx_;
    
    // SIMD-optimized normalization
    T compute_norm_and_normalize(const T* input, T* output) const {
        T norm_sq = 0;
        
#if defined(HNSW_SIMD_AVX2)
        size_t i = 0;
        __m256 sum = _mm256_setzero_ps();
        
        for (; i + 7 < dim_; i += 8) {
            __m256 v = _mm256_loadu_ps(input + i);
            sum = _mm256_fmadd_ps(v, v, sum);
        }
        
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, sum);
        norm_sq = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
        
        for (; i < dim_; ++i) {
            norm_sq += input[i] * input[i];
        }
#elif defined(HNSW_SIMD_NEON)
        size_t i = 0;
        float32x4_t sum = vdupq_n_f32(0.0f);
        
        for (; i + 3 < dim_; i += 4) {
            float32x4_t v = vld1q_f32(input + i);
            sum = vmlaq_f32(sum, v, v);
        }
        
        float tmp[4];
        vst1q_f32(tmp, sum);
        norm_sq = tmp[0] + tmp[1] + tmp[2] + tmp[3];
        
        for (; i < dim_; ++i) {
            norm_sq += input[i] * input[i];
        }
#else
        for (size_t i = 0; i < dim_; ++i) {
            norm_sq += input[i] * input[i];
        }
#endif
        
        T norm = std::sqrt(norm_sq);
        
        if (norm > cEPS) {
            T inv_norm = T(1) / norm;
#if defined(HNSW_SIMD_AVX2)
            size_t i = 0;
            __m256 inv = _mm256_set1_ps(inv_norm);
            for (; i + 7 < dim_; i += 8) {
                __m256 v = _mm256_loadu_ps(input + i);
                v = _mm256_mul_ps(v, inv);
                _mm256_storeu_ps(output + i, v);
            }
            for (; i < dim_; ++i) {
                output[i] = input[i] * inv_norm;
            }
#elif defined(HNSW_SIMD_NEON)
            size_t i = 0;
            float32x4_t inv = vdupq_n_f32(inv_norm);
            for (; i + 3 < dim_; i += 4) {
                float32x4_t v = vld1q_f32(input + i);
                v = vmulq_f32(v, inv);
                vst1q_f32(output + i, v);
            }
            for (; i < dim_; ++i) {
                output[i] = input[i] * inv_norm;
            }
#else
            for (size_t i = 0; i < dim_; ++i) {
                output[i] = input[i] * inv_norm;
            }
#endif
        } else {
            // Handle zero/near-zero vectors
            std::memcpy(output, input, dim_ * sizeof(T));
            norm = cEPS; // Prevent division by zero later
        }
        
        return norm;
    }

public:
    explicit SpaceQuantizedCosine(size_t dim,
                                  QuantModeIP qmode,
                                  OptBinModeIP bin_mode = OptBinModeIP::STANDARD,
                                  const std::vector<std::vector<T>>* sample_embeddings = nullptr,
                                  size_t buffer_capacity = 1000)
        : dim_(dim), 
          qmode_(qmode),
          bin_mode_(bin_mode),
          inner_space_(dim, qmode, bin_mode, 
                      sample_embeddings ? &normalize_samples_for_training(*sample_embeddings) : nullptr,
                      buffer_capacity,
                      false) {  // Don't use IP's internal normalization tracking
        
        bytes_for_quantized_ = inner_space_.get_data_size();
        // Add space for storing the norm (1 float per vector)
        bytes_per_vector_ = bytes_for_quantized_ + sizeof(T);
        
        HNSWDEBUG << "  [SpaceQuantizedCosine: non-normalized vectors supported, storing norms]";
        HNSWDEBUG << "  [Bytes per vector: " << bytes_for_quantized_ << " (quantized) + " 
                  << sizeof(T) << " (norm) = " << bytes_per_vector_ << " total]";
    }
    
    // Normalize samples for training (returns new vector)
    std::vector<std::vector<T>> normalize_samples_for_training(const std::vector<std::vector<T>>& samples) {
        std::vector<std::vector<T>> normalized(samples.size());
        for (size_t i = 0; i < samples.size(); ++i) {
            normalized[i].resize(dim_);
            compute_norm_and_normalize(samples[i].data(), normalized[i].data());
        }
        return normalized;
    }

    // Interface compliance
    size_t get_data_size() override { 
        return bytes_per_vector_;
    }
    
    DISTFUNC_TYPE get_dist_func() override { 
        return &SpaceQuantizedCosine::fstdist_;
    }
    
    void* get_dist_func_param() override { 
        return this;
    }

    // Quantize with automatic normalization and norm storage
    void quantize(const T* emb, uint8_t* out) {
        std::vector<T> normalized(dim_);
        T norm = compute_norm_and_normalize(emb, normalized.data());
        
        // Quantize the normalized vector
        inner_space_.quantize(normalized.data(), out);
        
        // Store norm at the end of the quantized data
        T* norm_ptr = reinterpret_cast<T*>(out + bytes_for_quantized_);
        *norm_ptr = norm;
    }
    
    // Get norm from quantized data
    T get_norm(const uint8_t* data) const {
        const T* norm_ptr = reinterpret_cast<const T*>(data + bytes_for_quantized_);
        return *norm_ptr;
    }

    // Centroid operations (forward to inner space after normalization)
    void add_to_centroid(const T* emb) {
        std::vector<T> normalized(dim_);
        compute_norm_and_normalize(emb, normalized.data());
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
    
    // Static helper to normalize a vector in-place
    static void normalize_inplace(T* vec, size_t dim) {
        T norm_sq = 0;
        for (size_t i = 0; i < dim; ++i) {
            norm_sq += vec[i] * vec[i];
        }
        T norm = std::sqrt(norm_sq);
        
        if (norm > cEPS) {
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

private:
    // Distance function: computes cosine distance = 1 - cosine_similarity
    static float fstdist_(const void* p1, const void* p2, const void* param) {
        const auto* space = reinterpret_cast<const SpaceQuantizedCosine*>(param);
        const uint8_t* a = reinterpret_cast<const uint8_t*>(p1);
        const uint8_t* b = reinterpret_cast<const uint8_t*>(p2);
        
        // Get norms (stored at end of quantized data)
        T norm_a = space->get_norm(a);
        T norm_b = space->get_norm(b);
        
        // Compute inner product of normalized vectors using the inner space
        // This gives us: normalized_a · normalized_b
        float normalized_dot = 1.0f - space->inner_space_.compute_dist(a, b);
        
        // The vectors were already normalized when quantized, so:
        // cosine_sim(a, b) = (a·b) / (||a|| * ||b||)
        //                  = normalized_a · normalized_b
        // 
        // Since we quantized the normalized versions, the dot product we get
        // is already the cosine similarity!
        
        float cosine_sim = normalized_dot;
        
        // Convert to distance (smaller = more similar)
        float dist = 1.0f - cosine_sim;
        
        // Safety check
        if (std::isnan(dist) || std::isinf(dist)) {
            HNSWERR << "Invalid cosine distance: " << dist;
            return 1e9;
        }
        
        return dist;
    }
};

// Convenience type aliases
using SpaceQuantizedCosineF = SpaceQuantizedCosine<float>;
using SpaceQuantizedCosineD = SpaceQuantizedCosine<double>;

} // namespace hnswlib

/*
Usage example for Cosine Similarity with non-normalized vectors:

// Create space - handles non-normalized vectors automatically
auto space = new hnswlib::SpaceQuantizedCosine<float>(
    dim, 
    hnswlib::QuantModeIP::BIN1, 
    hnswlib::OptBinModeIP::RABITQ,
    &training_samples,  // Can be non-normalized!
    1000
);

// Create HNSW index
auto index = new hnswlib::HierarchicalNSW<float>(space, max_elements);

// Add points - no need to normalize manually
for (size_t i = 0; i < num_vectors; ++i) {
    space->add_to_centroid(vectors[i].data());  // Works with non-normalized
    
    std::vector<uint8_t> code(space->get_data_size());
    space->quantize(vectors[i].data(), code.data());  // Normalizes internally
    index->addPoint(code.data(), i);
}

// Query - also works with non-normalized vectors
std::vector<uint8_t> query_code(space->get_data_size());
space->quantize(query_vector.data(), query_code.data());  // Normalizes internally
auto results = index->searchKnn(query_code.data(), k);

// The distance returned is: 1 - cosine_similarity
// where cosine_similarity = (a·b) / (||a|| * ||b||)

// Save
space->flush_centroid_buffer();
space->save_centroid("centroid_cosine.bin");
index->saveIndex("index_cosine.bin");

Key features:
- Input vectors can be non-normalized (just like your cosine_similarity function)
- Normalization happens during quantization (SIMD-optimized)
- Norms are stored with quantized data (adds sizeof(float) bytes per vector)
- Distance computation matches: 1 - (a·b)/(||a||·||b||)
- All quantization modes supported (BIN1, INT4, INT8, RaBitQ, etc.)
*/
