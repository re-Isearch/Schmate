#include "ModelPathStore.hpp"
#include <vector>

bool ModelPathStore::set(const std::string& key, const std::string& value) {
    if (key.empty() || value.empty()) {
        return false;
    }
    
    // Check capacity before adding new entry
    if (store.find(key) == store.end() && store.size() >= MAX_ENTRIES) {
        return false;
    }
    
    store[key] = value;
    return true;
}

std::optional<std::string> ModelPathStore::get(const std::string& key) const {
    auto it = store.find(key);
    if (it != store.end()) {
        return it->second;
    }
    return std::nullopt;
}

bool ModelPathStore::remove(const std::string& key) {
    return store.erase(key) > 0;
}

std::vector<std::string> ModelPathStore::keys() const {
    std::vector<std::string> result;
    result.reserve(store.size());
    for (const auto& pair : store) {
        result.push_back(pair.first);
    }
    return result;
}

bool ModelPathStore::save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;
    
    size_t count = store.size();
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    
    for (const auto& [key, value] : store) {
        size_t keyLen = key.size();
        size_t valueLen = value.size();
        
        file.write(reinterpret_cast<const char*>(&keyLen), sizeof(keyLen));
        file.write(key.c_str(), keyLen);
        
        file.write(reinterpret_cast<const char*>(&valueLen), sizeof(valueLen));
        file.write(value.c_str(), valueLen);
    }
    
    return file.good();
}

bool ModelPathStore::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;
    
    store.clear();
    
    size_t count;
    file.read(reinterpret_cast<char*>(&count), sizeof(count));
    
    if (count > MAX_ENTRIES) {
        return false;
    }
    
    for (size_t i = 0; i < count; i++) {
        size_t keyLen, valueLen;
        
        file.read(reinterpret_cast<char*>(&keyLen), sizeof(keyLen));
        std::string key(keyLen, '\0');
        file.read(&key[0], keyLen);
        
        file.read(reinterpret_cast<char*>(&valueLen), sizeof(valueLen));
        std::string value(valueLen, '\0');
        file.read(&value[0], valueLen);
        
        store[key] = value;
    }
    
    return file.good();
}

bool ModelPathStore::exportToText(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) return false;
    
    for (const auto& [key, value] : store) {
        file << key << "=" << value << "\n";
    }
    
    return file.good();
}

bool ModelPathStore::importFromText(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) return false;
    
    store.clear();
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            
            // Trim whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            if (!set(key, value)) {
                return false;
            }
        }
    }
    
    return true;
}
