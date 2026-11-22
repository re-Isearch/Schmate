#include "unified_hnsw.hpp"
#include "LRUCache.hpp"
#include "BertIndex.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

using namespace hnswlib;

static void require(bool ok, const char* message)
{
    if (!ok) { std::cerr << message << '\n'; std::exit(1); }
}

static std::map<labeltype, std::vector<float>> snapshot(LSMVectorStorage& store)
{
    std::map<labeltype, std::vector<float>> values;
    store.for_each_vector([&](labeltype label, const float* vector) {
        require(values.emplace(label, std::vector<float>(vector, vector + 16)).second,
                "vector enumeration must emit each label once");
    });
    return values;
}

static void storage_regressions(bool streaming)
{
    VectorStorageConfig config;
    config.mode = VectorStorageMode::IN_MEMORY;
    config.flush_threshold = 1000;
    config.compact_threshold = 1000;
    config.use_streaming_compaction = streaming;
    const std::string base = streaming ? "streaming" : "in-memory";
    LSMVectorStorage store(16, config);
    store.set_basename(base);
    std::vector<float> first(16), second(16), latest(16), other(16, -7);
    for (size_t i = 0; i < 16; ++i) {
        first[i] = float(i + 1);
        second[i] = float(i + 2);
        latest[i] = float(i + 3);
    }
    store.addPoint(7, first.data()); store.addPoint(11, other.data());
    require(store.flush_delta(), "first vector delta");
    store.addPoint(7, second.data());
    require(store.flush_delta(), "second vector delta");
    store.addPoint(7, latest.data());
    const auto expected = snapshot(store);
    require(expected.size() == 2 && expected.at(7) == latest, "latest active update wins");

    const std::string path = base + ".saved";
    std::ofstream output(path, std::ios::binary);
    require(store.save_vectors_to_stream(output), "serialize original vectors");
    output.close();
    std::ifstream input(path, std::ios::binary);
    LSMVectorStorage restored(16, config);
    require(restored.load_vectors(path, input, VectorStorageMode::IN_MEMORY), "load vector snapshot");
    require(snapshot(restored) == expected, "all vector coordinates survive serialization");

    require(store.compact(), "compact newest vector values");
    require(snapshot(store) == expected, "compaction retains newest values once");
    require(store.with_vector(7, [&](const float* vector) {
        require(std::equal(latest.begin(), latest.end(), vector), "compacted lookup");
    }), "compacted vector remains readable");
    store.addPoint(7, first.data());
    require(store.flush_delta(), "delta after compaction");
    store.addPoint(7, latest.data());
    require(store.flush_delta() && store.compact(), "compact multiple updates over base");
    require(snapshot(store) == expected, "newest flushed override wins over base");
    store.clear();
    require(snapshot(store).empty(), "clear removes pending and compacted vectors");
}

static UnifiedIndexMeta float_meta()
{
    UnifiedIndexMeta meta(2, 16, Metric::L2, QuantMode::NONE, OptBinMode::STANDARD);
    meta.flush_threshold_ = 1;
    meta.storage_mode_ = VectorStorageMode::DISABLED;
    return meta;
}

static void delta_replay_regressions(VectorStorageMode mode)
{
    const std::string path = std::filesystem::absolute(
        mode == VectorStorageMode::IN_MEMORY ? "replay-memory" : "replay-mapped").string();
    const size_t zero = 0;
    { std::ofstream out(path, std::ios::binary);
      out.write(reinterpret_cast<const char*>(&zero), sizeof(zero)); }
    VectorStorageConfig config;
    config.mode = mode;
    LSMVectorStorage original(16, config);
    original.set_basename(path);
    const std::vector<float> first(16, 1), second(16, 2), third(16, 3);
    original.addPoint(1, first.data()); require(original.flush_delta(), "first replay delta");
    original.addPoint(1, second.data()); require(original.flush_delta(), "second replay delta");
    LSMVectorStorage restored(16, config);
    std::ifstream input(path, std::ios::binary);
    require(restored.load_vectors(path, input, mode), "replay deltas with an absolute basename");
    require(snapshot(restored).at(1) == second, "newest replayed update wins");
    restored.addPoint(1, third.data());
    require(restored.flush_delta() && std::filesystem::exists(path + ".delta.03"),
            "restart retains highest delta version");
    restored.clear();
}

static bool has_label(UnifiedIndex& index, const float* query, labeltype label)
{
    for (const auto& result : index.searchKnnCloserFirst(query, 16, false))
        if (result.second == label) return true;
    return false;
}

static void cache_budget_regressions()
{
    struct Sample { size_t bytes; };
    LRUCache<int, Sample> cache(100, nullptr, 100,
        [](const std::shared_ptr<Sample>& item) { return item->bytes; });
    cache.put(1, std::make_shared<Sample>(Sample{60}));
    cache.put(2, std::make_shared<Sample>(Sample{60}));
    require(!cache.contains(1) && cache.contains(2),
            "byte-budget LRU evicts oldest index");
    cache.put(3, std::make_shared<Sample>(Sample{180}));
    require(cache.size() == 1 && cache.contains(3),
            "oversized active graph remains cached alone");
    cache.put(4, std::make_shared<Sample>(Sample{20}));
    require(cache.size() == 1 && cache.contains(4),
            "smaller graph replaces oversized entry");
}

static void search_only_capacity_regressions()
{
    UnifiedIndexMeta meta = float_meta();
    meta.max_elements_ = 500000; // Sharding limit, not the search working set.
    const float a[]{1, 0}, b[]{0, 1};
    UnifiedIndex writer(meta);
    writer.addPoint(a, 1);
    writer.addPoint(b, 2);
    require(writer.saveIndex("search-only-capacity.hnsw"), "save sparse HNSW graph");

    UnifiedIndex reader(meta);
    require(reader.loadIndex("search-only-capacity.hnsw", true), "load sparse graph for search");
    require(reader.size() == 2, "search-only load preserves element count");
    require(reader.get_meta().max_elements_ == 2, "search-only load uses populated capacity");
    require(has_label(reader, a, 1) && has_label(reader, b, 2), "search-only graph remains searchable");

    UnifiedIndex writable(meta);
    require(writable.loadIndex("search-only-capacity.hnsw", false), "writable load");
    require(writable.get_meta().max_elements_ == meta.max_elements_,
            "writable load retains shard capacity");
}

static void dirty_state_regressions()
{
    const float a[]{1, 0}, b[]{0, 1};
    UnifiedIndex index(float_meta());
    index.addPoint(a, 1); index.addPoint(b, 2);
    require(index.saveIndex("dirty.hnsw") && std::filesystem::exists("dirty.hnsw"),
            "automatic flush before assigning a path must preserve dirty state");
    index.markDelete(1);
    require(index.saveIndex("dirty.hnsw"), "save deletion-only mutation");
    UnifiedIndex restored(float_meta());
    require(restored.loadIndex("dirty.hnsw"), "reload deleted index");
    require(!has_label(restored, a, 1) && has_label(restored, a, 2), "deletion survives reload");
    restored.unmarkDelete(1);
    require(restored.saveIndex("dirty.hnsw"), "save undelete-only mutation");
    UnifiedIndex undeleted(float_meta());
    require(undeleted.loadIndex("dirty.hnsw") && has_label(undeleted, a, 1), "undelete survives reload");
    require(undeleted.updateDeletedElements([](labeltype label) { return label == 2; }) == 1,
            "bulk deletion callback");
    require(undeleted.saveIndex("dirty.hnsw"), "save bulk deletion");
    UnifiedIndex bulk(float_meta());
    require(bulk.loadIndex("dirty.hnsw") && !has_label(bulk, a, 2), "bulk deletion survives reload");
    require(bulk.saveIndex("copy.hnsw") && std::filesystem::exists("copy.hnsw"),
            "save clean index to another path");
    bulk.unmarkDelete(2);
    require(bulk.saveIndex("copy.hnsw") && bulk.flush(), "save-as retains primary dirty state");
    UnifiedIndex primary(float_meta());
    require(primary.loadIndex("dirty.hnsw") && has_label(primary, b, 2),
            "flush persists deletion-only changes to the primary path after save-as");

    UnifiedIndex raw(float_meta());
    raw.rawAddPoint(a, 9);
    require(!raw.saveIndex("missing-directory/index.hnsw"), "failed save reports failure");
    require(raw.saveIndex("raw.hnsw"), "failed save retains dirty state for retry");
    require(raw.get_filepath() == "raw.hnsw", "failed first save must not bind an unusable path");
    UnifiedIndex raw_restored(float_meta());
    require(raw_restored.loadIndex("raw.hnsw") && has_label(raw_restored, a, 9), "raw addition persists");
}

static void mapped_rescoring_regressions()
{
    UnifiedIndexMeta meta(16, 16, Metric::Cosine, QuantMode::INT8, OptBinMode::STANDARD,
                          StorageType::INT8, true, 4, 40);
    meta.storage_mode_ = VectorStorageMode::MEMORY_MAPPED;
    const std::vector<float> first{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    auto second = first; second[0] = 0; second[1] = 1;
    UnifiedIndex original(meta);
    original.addPoint(first.data(), 1); original.addPoint(second.data(), 2);
    require(original.saveIndex("mapped.hnsw"), "save rescoring index");
    UnifiedIndex mapped(meta);
    require(mapped.loadIndex("mapped.hnsw"), "load mapped original vectors");
    require(mapped.searchKnnCloserFirst(first.data(), 2, true).front().second == 1, "mapped rescoring");
    mapped.addPoint(second.data(), 1);
    require(mapped.saveIndex("mapped.hnsw"), "save while old vector region remains mapped");
    // Sidecars from an older checkpoint must not replace the paired graph's
    // freshly saved vector region or move its input stream to the wrong offset.
    for (const char* stale : {"mapped.hnsw.delta.07", "mapped.hnsw.vectors.00001"}) {
        std::ofstream out(stale, std::ios::binary);
        const size_t count = 1;
        const labeltype label = 1;
        out.write(reinterpret_cast<const char*>(&count), sizeof(count));
        out.write(reinterpret_cast<const char*>(&label), sizeof(label));
        out.write(reinterpret_cast<const char*>(first.data()), first.size() * sizeof(float));
    }
    UnifiedIndex updated(meta);
    require(updated.loadIndex("mapped.hnsw"), "reload overwritten mapped index");
    const auto results = updated.searchKnnCloserFirst(first.data(), 2, true);
    require(results.size() == 2, "updated index retains labels");
    for (const auto& [distance, label] : results)
        require(std::abs(distance - 1.0f) < 0.0001f, "updated vectors survive mapped save");
    require(mapped.loadIndex("mapped.hnsw"), "reload into an index with existing vector caches");
    for (const auto& [distance, label] : mapped.searchKnnCloserFirst(first.data(), 2, true))
        require(std::abs(distance - 1.0f) < 0.0001f, "reload discards older in-process caches");
}

static void rescore_metric_regressions()
{
    const float query[]{1, 0}, larger[]{2, 0}, far[]{4, 4};
    UnifiedIndexMeta meta(2, 16, Metric::IP, QuantMode::INT8, OptBinMode::STANDARD,
                          StorageType::INT8, true, 4, 40);
    meta.storage_mode_ = VectorStorageMode::IN_MEMORY;
    UnifiedIndex ip(meta);
    ip.addPoint(query, 1); ip.addPoint(larger, 2);
    const auto products = ip.searchKnnCloserFirst(query, 2, true);
    require(products.size() == 2 && products.front().second == 2 &&
            std::abs(products.front().first + 1.0f) < 0.0001f, "IP rescoring preserves vector magnitude");
    meta.metric_ = Metric::L2;
    UnifiedIndex l2(meta);
    l2.addPoint(far, 1);
    require(std::abs(l2.searchKnnCloserFirst(query, 1, true).front().first - 25.0f) < 0.0001f,
            "L2 rescoring retains squared-distance units");
}


static void deferred_insertion_regressions()
{
    SBertGGML embedder(2); // Raw vector test: no model required.
    HnswConfig cfg;
    cfg.max_elements = 8;
    cfg.flush_threshold = 1; // Deferred entries must survive automatic flush threshold.

    for (const auto policy : {InsertionPolicy::Immediate, InsertionPolicy::Deferred,
                              InsertionPolicy::Adaptive}) {
        const std::string base = policy == InsertionPolicy::Immediate ? "points-immediate" :
                                 policy == InsertionPolicy::Deferred  ? "points-deferred" :
                                                                       "points-adaptive";
        BertIndex::unlink(base);
        {
            BertIndex graph(embedder, cfg, base, false, nullptr, policy);
            graph.set_insertion_policy(policy);
            if (policy == InsertionPolicy::Adaptive) graph.set_adaptive_defer(true);
            graph.append("[1.0, 0.0]", 100);
            graph.append("[0.0, 1.0]", 101);
            if (policy != InsertionPolicy::Immediate) {
                require(!graph.graph_resident(), "deferred insertion never allocated a graph");
                require(graph.pending_vectors() == 2, "deferred queue holds both labels");
                require(graph.size() == 0, "deferred vectors not yet in HNSW");
            }
            graph.flush_deferred();
            require(graph.pending_vectors() == 0, "flush drains deferred queue");
            require(graph.size() == 2, "both vectors inserted after flush");
            if (policy != InsertionPolicy::Immediate)
                require(!graph.graph_resident(), "first replay released graph memory");
            require(graph.label_count() == 2, "original labels retained");
            require(graph.get_sentence_id(0) == 100 && graph.get_sentence_id(1) == 101,
                    "offset mapping retained during replay");
            if (policy != InsertionPolicy::Immediate) {
                // A released graph must be reloaded, extended and saved again.
                graph.append("[1.0, 1.0]", 102);
                graph.append("[2.0, 1.0]", 103);
                require(graph.pending_vectors() == 2, "second batch queued without graph");
                require(!graph.graph_resident(), "repeated appends do not load graph");
                graph.flush_deferred();
                require(graph.size() == 4, "second batch extends the original graph");
                require(graph.pending_vectors() == 0, "second flush drains the queue");
                require(!graph.graph_resident(), "second replay released graph memory");
                require(graph.get_sentence_id(2) == 102 && graph.get_sentence_id(3) == 103,
                        "second batch preserves new offset entries");
            }
        }
        if (policy != InsertionPolicy::Immediate) {
            BertIndex resumed(embedder, cfg, base, false, nullptr, policy);
            require(!resumed.graph_resident(), "deferred writable reopen does not load HNSW");
            require(resumed.size() == 4, "deferred reopen retains population");
            resumed.append("[3.0, 1.0]", 104);
            require(!resumed.graph_resident(), "deferred append after reopen stays disk-only");
            resumed.flush_deferred();
            require(!resumed.graph_resident() && resumed.size() == 5,
                    "third batch replayed and released");
        }
        {
            BertIndex restored(embedder, cfg, base, true);
            const size_t expected = policy == InsertionPolicy::Immediate ? 2 : 5;
            require(restored.size() == expected, "all deferred batches survive reload");
            require(restored.get_sentence_id(0) == 100 && restored.get_sentence_id(1) == 101,
                    "original labels and offsets survive reload");
            if (expected == 5) {
                require(restored.get_sentence_id(2) == 102 && restored.get_sentence_id(3) == 103,
                        "second batch survives reload");
                require(restored.get_sentence_id(4) == 104, "third batch survives reload");
            }
        }
    }
}

int main()
{
    storage_regressions(true);
    storage_regressions(false);
    delta_replay_regressions(VectorStorageMode::IN_MEMORY);
    delta_replay_regressions(VectorStorageMode::MEMORY_MAPPED);
    cache_budget_regressions();
    deferred_insertion_regressions();
    search_only_capacity_regressions();
    dirty_state_regressions();
    mapped_rescoring_regressions();
    rescore_metric_regressions();
    std::cout << "Vector persistence regressions passed\n";
}
