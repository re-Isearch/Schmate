#include "ShardedIndex.hpp"
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <mutex>

using namespace hnswlib;

// The standalone application supplies this integration hook. These tests use
// the real file store and never attach an IB bridge.
std::unique_ptr<SentenceStore> SentenceStoreFactory::CreateBridgeStore(void*)
{
    throw std::runtime_error("IB bridge is not used by standalone search tests");
}

static void require(bool ok, const char* message)
{
    if (!ok) { std::cerr << message << '\n'; std::exit(1); }
}

// Deterministic embeddings exercise the real search and persistence layers
// without requiring a model download or changing the BERT backend.
struct FixtureEmbedder : SBertGGML {
    std::atomic<size_t> calls{0};
    std::mutex mutex;
    std::string last_input;
    FixtureEmbedder() : SBertGGML(size_t(2)) { name = "e5-energy-test"; arch = "bert"; }
    std::vector<float> encode_text(const std::string& text, bool = false) override {
        ++calls;
        { std::lock_guard<std::mutex> lock(mutex); last_input = text; }
        if (text.ends_with("middle")) return {0.6f, 0.8f};
        if (text.ends_with("far")) return {0, 1};
        if (text.ends_with("bad")) return {1};
        return {1, 0};
    }
    std::vector<std::vector<float>> encode_batch(
        const std::vector<std::string>& texts, bool debug = false) override {
        return BaseEmbedder::encode_batch(texts, debug);
    }
};

static HnswConfig config(Metric metric)
{
    HnswConfig cfg;
    cfg.set_metric(metric);
    cfg.max_elements = 16;
    cfg.M = 4;
    cfg.ef_construction = 40;
    cfg.default_k = 3;
    cfg.flush_threshold = 10000;
    cfg.lock_on_append = false;
    cfg.min_candidates = 1;
    return cfg;
}

static void score_regressions(Metric metric)
{
    FixtureEmbedder embedder;
    auto cfg = config(metric);
    BertIndex index(embedder, cfg, metric == Metric::L2 ? "l2" : "cosine");
    require(index.relative("question").empty(), "empty relative search");
    index.append("near", 101); index.append("middle", 102); index.append("far", 103);
    require(embedder.last_input == "passage: far", "document prefix");
    auto results = index.knn("question", 3);
    require(results.size() == 3 && results[0].sentence_id == 101 &&
            results[1].sentence_id == 102 && results[2].sentence_id == 103,
            "all public scores sort best first for the actual index metric");
    require(results[0].score > results[1].score && results[1].score > results[2].score,
            "descending score order");
    require(embedder.last_input == "query: question", "query prefix");
    results = index.radius("question", 0.9f);
    require(results.size() == 1 && results.front().sentence_id == 101, "radius is a minimum score");
    embedder.calls = 0;
    results = index.relative("question", metric == Metric::L2 ? 0.5f : 0.7f, 3);
    require(results.size() == 2 && results.front().sentence_id == 101, "relative score threshold");
    require(embedder.calls == 1, "relative search embeds and traverses once");
    embedder.calls = 0;
    results = index.adaptive("question", 0.9f, 1, 3, 1.0f);
    require(results.size() == 1 && results.front().sentence_id == 101, "adaptive alpha threshold");
    require(embedder.calls == 1, "adaptive search embeds and traverses once");
    results = index.adaptive("question", 0.9f, 2, 3, 1.0f);
    require(results.size() == 2, "adaptive minimum result count");
    embedder.calls = 0;
    require(index.knn("bad", 3).empty(), "reject mismatched query embedding dimensions");
    require(index.knn("   ", 3).empty() && embedder.calls == 1, "skip invalid queries");
    embedder.calls = 0;
    index.epsilon_search("question", 1.0f);
    require(embedder.calls == 1 && embedder.last_input == "query: question", "epsilon query prefix");
    index.flush();
}

static void shard_regressions()
{
    FixtureEmbedder embedder;
    auto cfg = config(Metric::L2);
    cfg.max_elements = 2;
    ShardedIndex index(embedder, cfg, "shards");
    index.append("near", 201); index.append("middle", 202); index.append("far", 203);
    require(index.shard_count() == 2, "shard rollover");
    const auto results = index.knn("question", 3);
    require(results.size() == 3 && results[0].sentence_id == 201 && results[2].shard == 1,
            "rolled-over shards retain matching search tuners");
    index.clear();
    require(index.shard_count() == 0 && index.knn("question").empty(), "clear shard and tuner state");
}

int main()
{
    score_regressions(Metric::L2);
    score_regressions(Metric::Cosine);
    shard_regressions();
    std::cout << "Search energy regressions passed\n";
}
