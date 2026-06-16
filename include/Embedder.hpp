#pragma once
#include <string>
#include <vector>

class BaseEmbedder {
public:
    virtual ~BaseEmbedder() = default;
    virtual std::vector<float> encode_text(const std::string &text, bool debug=false) = 0;

    // Batch encode. Default impl calls encode_text sequentially;
    // backends with native batch support should override.
    virtual std::vector<std::vector<float>>
    encode_batch(const std::vector<std::string> &texts, bool debug = false)
    {
        std::vector<std::vector<float>> results;
        results.reserve(texts.size());
        for (const auto &t : texts)
            results.push_back(encode_text(t, debug));
        return results;
    }

    virtual size_t embedding_dim() const = 0;
    virtual size_t Matryoshka_dim() const = 0;

    // Returns the exact GGUF architecture string (e.g., "bert", "nomic-bert-moe")
    virtual std::string_view model_architecture() const = 0;
    
    // Returns the model name or family identifier if needed for finer rules (e.g., "bge-large")
    virtual std::string_view model_name() const = 0;
};


