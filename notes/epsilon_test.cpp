#include "hnswlib/hnswlib.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <limits>

// Helper function to check vector validity
bool isVectorValid(const std::vector<float>& vec) {
    for (float val : vec) {
        if (std::isnan(val) || std::isinf(val)) {
            std::cout << "ERROR: Vector contains NaN or Inf!\n";
            return false;
        }
    }
    return true;
}

// Helper function to normalize vector
void normalizeVector(std::vector<float>& vec) {
    float norm = 0.0f;
    for (float val : vec) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    
    if (norm < 1e-10f) {
        std::cout << "WARNING: Vector has near-zero norm!\n";
        return;
    }
    
    for (float& val : vec) {
        val /= norm;
    }
}

// Verify vector is normalized
bool isNormalized(const std::vector<float>& vec, float tolerance = 0.01f) {
    float norm_sq = 0.0f;
    for (float val : vec) {
        norm_sq += val * val;
    }
    float norm = std::sqrt(norm_sq);
    return std::abs(norm - 1.0f) < tolerance;
}

// Manual inner product calculation for verification
float manualInnerProduct(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

int main() {
    int dim = 128;
    int max_elements = 100;
    
    std::cout << "=== Debugging InnerProduct Distance ===\n\n";
    
    // Create index
    hnswlib::InnerProductSpace space(dim);
    hnswlib::HierarchicalNSW<float>* alg = 
        new hnswlib::HierarchicalNSW<float>(&space, max_elements, 16, 200);
    
    // Create and add test vectors
    std::vector<std::vector<float>> test_vectors;
    
    for (int i = 0; i < 10; i++) {
        std::vector<float> vec(dim);
        
        // Generate random vector
        for (int d = 0; d < dim; d++) {
            vec[d] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        }
        
        // Check validity
        if (!isVectorValid(vec)) {
            std::cout << "Vector " << i << " is invalid!\n";
            continue;
        }
        
        // Normalize
        normalizeVector(vec);
        
        // Verify normalization
        if (!isNormalized(vec)) {
            std::cout << "WARNING: Vector " << i << " failed normalization check\n";
        }
        
        std::cout << "Adding vector " << i << ", norm = " 
                  << std::sqrt(manualInnerProduct(vec, vec)) << "\n";
        
        test_vectors.push_back(vec);
        alg->addPoint(vec.data(), i);
    }
    
    std::cout << "\n=== Testing Search ===\n";
    
    // Use first vector as query
    std::vector<float> query = test_vectors[0];
    
    std::cout << "Query vector norm: " 
              << std::sqrt(manualInnerProduct(query, query)) << "\n\n";
    
    // Test 1: Regular kNN search
    std::cout << "Test 1: Regular kNN search (k=5)\n";
    auto knn_results = alg->searchKnn(query.data(), 5);
    
    while (!knn_results.empty()) {
        auto [dist, label] = knn_results.top();
        knn_results.pop();
        
        // Manual verification
        float manual_dot = manualInnerProduct(query, test_vectors[label]);
        float manual_dist = 1.0f - manual_dot;
        
        std::cout << "Label: " << label 
                  << ", HNSWlib dist: " << dist
                  << ", Manual dist: " << manual_dist
                  << ", Dot product: " << manual_dot << "\n";
        
        // Check for anomalies
        if (std::abs(dist) > 10.0f || std::isnan(dist) || std::isinf(dist)) {
            std::cout << "  ^^^ ANOMALY DETECTED!\n";
        }
    }
    
    // Test 2: Epsilon search with stop condition
    std::cout << "\nTest 2: Epsilon search with stop condition\n";
    
    // CRITICAL: For InnerProduct, epsilon is the DISTANCE threshold
    // Distance = 1 - dot_product
    // For normalized vectors wanting cosine similarity > 0.5:
    // Distance < 1 - 0.5 = 0.5
    float epsilon_threshold = 0.5f;  // Find vectors with distance < 0.5
    
    std::cout << "Epsilon threshold: " << epsilon_threshold << "\n";
    
    hnswlib::EpsilonSearchStopCondition<float> stop_condition(
        epsilon_threshold, 10, max_elements);
    
    try {
        auto results = alg->searchStopConditionClosest(query.data(), stop_condition);
        
        std::cout << "Found " << results.size() << " results\n";
        
        for (const auto& [dist, label] : results) {
            // Manual verification
            float manual_dot = manualInnerProduct(query, test_vectors[label]);
            float manual_dist = 1.0f - manual_dot;
            
            std::cout << "Label: " << label 
                      << ", HNSWlib dist: " << dist
                      << ", Manual dist: " << manual_dist
                      << ", Cosine sim: " << (1.0f - dist) << "\n";
            
            // Check for anomalies
            if (std::abs(dist) > 10.0f || std::isnan(dist) || std::isinf(dist)) {
                std::cout << "  ^^^ ANOMALY DETECTED! dist = " << dist << "\n";
                std::cout << "  This indicates a serious bug!\n";
                
                // Additional diagnostics
                std::cout << "  Vector " << label << " first 5 values: ";
                for (int i = 0; i < 5; i++) {
                    std::cout << test_vectors[label][i] << " ";
                }
                std::cout << "\n";
            }
        }
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << "\n";
    }
    
    // Test 3: Self-search (should return distance ~0)
    std::cout << "\nTest 3: Self-search (query is in index)\n";
    auto self_results = alg->searchKnn(query.data(), 1);
    auto [self_dist, self_label] = self_results.top();
    std::cout << "Self-search distance: " << self_dist 
              << " (should be ~0.0)\n";
    
    if (std::abs(self_dist) > 1e-5) {
        std::cout << "WARNING: Self-search distance is not near zero!\n";
    }
    
    delete alg;
    
    std::cout << "\n=== Recommendations ===\n";
    std::cout << "1. Ensure all vectors are properly normalized before adding\n";
    std::cout << "2. Check for NaN/Inf values in your data\n";
    std::cout << "3. For InnerProduct with normalized vectors:\n";
    std::cout << "   - Distance range: [0, 2]\n";
    std::cout << "   - Distance = 1 - cosine_similarity\n";
    std::cout << "   - Use epsilon < 1.0 for similar items\n";
    std::cout << "4. If you see large negative values like -1e34:\n";
    std::cout << "   - Memory corruption or uninitialized data\n";
    std::cout << "   - Dimension mismatch\n";
    std::cout << "   - Bug in HNSWlib version (update to latest)\n";
    
    return 0;
}
