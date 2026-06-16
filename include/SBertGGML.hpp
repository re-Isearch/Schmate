#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include "Util.hpp"
#include "Logger.hpp"
#include "Embedder.hpp"

// Support now the new bert.cpp which in turn uses the latest ggml (0.12)

// Forward declaration of bert C API
//extern "C" {
#include "bert.h"
//} ;

#define NEW_BERT_API 1


struct SBertGGML : public BaseEmbedder {
    bert_ctx * ctx = nullptr;
    int n_embd = 0;
    std::string name;
    std::string arch;

    SBertGGML(const std::string & model_path, size_t threads = 4);

    std::string_view model_architecture() const override { return arch; }
    std::string_view model_name() const override { return name; }

    size_t embedding_dim() const override { return (size_t)n_embd; }
    size_t Matryoshka_dim() const override { return 0; }


    ~SBertGGML() {
        if (ctx) bert_free(ctx);
    }

#if NEW_BERT_API
  std::vector<float> encode_text(const std::string & text, bool debug=false) override {
    if (text.empty()) return std::vector<float>(n_embd, 0.0f);

    // This ensures only one thread uses the ggml backend at a time
    std::lock_guard<std::mutex> lock(encode_mutex);

    // 1. Tokenize using the new API 
    int32_t N = bert_n_max_tokens(ctx);
    bert_tokens tokens = bert_tokenize(ctx, text.c_str(), N);

    if (tokens.empty()) return std::vector<float>(n_embd, 0.0f);

    // 2. Wrap tokens in a batch of size 1
    bert_batch batch;
    batch.push_back(tokens);

    // 3. Prepare output buffer
    std::vector<float> emb(n_embd);
    int n_threads = calculate_optimal_threads();

    // 4. Use the new forward_batch call
    bert_forward_batch(ctx, batch, emb.data(), n_threads);

    return emb;
  }



  std::vector<std::vector<float>> encode_batch(const std::vector<std::string> &texts,
             bool debug = false) override
  {
    std::vector<std::vector<float>> results(texts.size(),
                                            std::vector<float>(n_embd, 0.0f));
    if (texts.empty()) return results;

    std::lock_guard<std::mutex> lock(encode_mutex);

    bert_batch batch;
    std::vector<size_t> valid_indices;   // maps batch slot → results slot

    int32_t N = bert_n_max_tokens(ctx);
    for (size_t i = 0; i < texts.size(); ++i) {
        if (texts[i].empty()) continue;
        bert_tokens toks = bert_tokenize(ctx, texts[i].c_str(), N);
        if (toks.empty()) continue;
        batch.push_back(std::move(toks));
        valid_indices.push_back(i);
    }

    if (batch.empty()) return results;

    // One backend call for the whole batch.
    // bert_forward_batch writes embeddings contiguously: slot 0, slot 1, …
    std::vector<float> flat(n_embd * batch.size());
    int n_threads = calculate_optimal_threads();
    bert_forward_batch(ctx, batch, flat.data(), n_threads);

    for (size_t slot = 0; slot < valid_indices.size(); ++slot) {
        size_t ri = valid_indices[slot];
        float *src = flat.data() + slot * n_embd;
        results[ri].assign(src, src + n_embd);
    }
    return results;
   }

   void encode(const char ** texts, float ** embeddings, int n_inputs);
   void encode(const char * text, float * embedding);
#else
   std::vector<float> encode_text(const std::string & text, bool debug=false) override {
        const int MAX_TOKENS = 512;
        std::vector<bert_vocab_id> tokens(MAX_TOKENS);
        int32_t n_tokens = 0; // was 512;
	// std::cerr << "Encoding: \"" << text << "\"\n";
        bert_tokenize(ctx, text.c_str(), tokens.data(), &n_tokens, (int)tokens.size());

	// new: check error
        if (n_tokens <= 0) return std::vector<float>(n_embd, 0.0f);

        std::vector<float> emb(n_embd);
        this->eval(tokens.data(), n_tokens, emb.data());
        return emb;
    }

   void eval (bert_vocab_id * tokens, int32_t n_tokens, float * embeddings);
   void encode(const char ** texts, float ** embeddings, int n_inputs, int batch_size = 1);
   void encode( const char * texts, float * embeddings, int batch_size = 1);
#endif

   void reset_context() {
     bert_free(ctx);
     ctx = nullptr;
   }

private:
   static int calculate_optimal_threads( int hardware_threads = 0, int batch_size = 1, bool is_cpu_only = true,
	double cpu_utilization_target = 0.85);
   size_t _threads;
   std::mutex encode_mutex; // The hardware gatekeeper
};


