#pragma once

/* Hash Based */
#ifndef HF_MODEL_MAPPER_H
#define HF_MODEL_MAPPER_H

#include <string>
#include <array>
#include <optional>
#include <vector>
#include <fstream>
#include <cstring>

// Fast hash-based key-value store for HuggingFace model mappings
class HFModelMapper {
public:
    static constexpr size_t MAX_ENTRIES = 100;
    static constexpr size_t TABLE_SIZE = 151; // Prime number > MAX_ENTRIES for better distribution
    static constexpr size_t KEY_SIZE = 64;
    static constexpr size_t REPO_SIZE = 256;
    static constexpr size_t FILENAME_SIZE = 256;
    static constexpr size_t URL_SIZE = 512;
    
    struct Entry {
        char key[KEY_SIZE];
        char repo[REPO_SIZE];
        char filename[FILENAME_SIZE];
        char downloadURL[URL_SIZE];
        bool active;
        
        Entry() : key{0}, repo{0}, filename{0}, downloadURL{0}, active(false) {}
    };
    
    std::array<Entry, TABLE_SIZE> table;  // Hash table
    
private:
    size_t count;
    
    // Simple hash function (FNV-1a)
    size_t hash(const std::string& key) const {
        size_t h = 2166136261u;
        for (char c : key) {
            h ^= static_cast<unsigned char>(c);
            h *= 16777619u;
        }
        return h % TABLE_SIZE;
    }
    
    // Find entry index (returns TABLE_SIZE if not found)
    size_t findIndex(const std::string& key) const {
        size_t idx = hash(key);
        size_t probes = 0;
        
        while (probes < TABLE_SIZE) {
            if (!table[idx].active) {
                return TABLE_SIZE; // Not found
            }
            if (std::strcmp(table[idx].key, key.c_str()) == 0) {
                return idx; // Found
            }
            idx = (idx + 1) % TABLE_SIZE; // Linear probing
            probes++;
        }
        return TABLE_SIZE; // Not found
    }
    
    // Find empty slot or existing key
    size_t findSlot(const std::string& key) const {
        size_t idx = hash(key);
        size_t probes = 0;
        
        while (probes < TABLE_SIZE) {
            if (!table[idx].active || std::strcmp(table[idx].key, key.c_str()) == 0) {
                return idx;
            }
            idx = (idx + 1) % TABLE_SIZE; // Linear probing
            probes++;
        }
        return TABLE_SIZE; // Table full
    }
    
public:
    HFModelMapper();
    
    // Set a mapping with all fields
    bool set(const std::string& key, const std::string& repo, 
             const std::string& filename, const std::string& downloadURL = "");
    
    // Get individual fields by key - O(1) average case
    std::optional<std::string> getRepo(const std::string& key) const;
    std::optional<std::string> getFilename(const std::string& key) const;
    std::optional<std::string> getDownloadURL(const std::string& key) const;
    
    // Remove a key
    bool remove(const std::string& key);
    
    // Get all keys
    std::vector<std::string> keys() const;
    
    // Get count of active entries
    size_t size() const { return count; }
    
    // Clear all entries
    void clear();
    
    // Save to binary file
    bool save(const std::string& filename) const;
    
    // Load from binary file
    bool load(const std::string& filename);
    
    // Export to text file (key|repo|filename|downloadURL format)
    bool exportToText(const std::string& filename) const;
    
    // Import from text file
    bool importFromText(const std::string& filename);
};

#endif // HF_MODEL_MAPPER_H
