#include "HFModelMapper.hpp"
#include <cstring>

HFModelMapper::HFModelMapper() : count(0) {
    for (auto& e : entries) {
        e.active = false;
    }
}

bool HFModelMapper::set(const std::string& key, const std::string& repo, 
                        const std::string& filename, const std::string& downloadURL) {
    if (key.empty() || key.size() >= KEY_SIZE || 
        repo.size() >= REPO_SIZE || filename.size() >= FILENAME_SIZE ||
        downloadURL.size() >= URL_SIZE) {
        return false;
    }
    
    // Check if key exists
    for (auto& e : entries) {
        if (e.active && std::strcmp(e.key, key.c_str()) == 0) {
            std::strncpy(e.repo, repo.c_str(), REPO_SIZE - 1);
            e.repo[REPO_SIZE - 1] = '\0';
            std::strncpy(e.filename, filename.c_str(), FILENAME_SIZE - 1);
            e.filename[FILENAME_SIZE - 1] = '\0';
            std::strncpy(e.downloadURL, downloadURL.c_str(), URL_SIZE - 1);
            e.downloadURL[URL_SIZE - 1] = '\0';
            return true;
        }
    }
    
    // Add new entry
    if (count >= MAX_ENTRIES) {
        return false;
    }
    
    for (auto& e : entries) {
        if (!e.active) {
            std::strncpy(e.key, key.c_str(), KEY_SIZE - 1);
            e.key[KEY_SIZE - 1] = '\0';
            std::strncpy(e.repo, repo.c_str(), REPO_SIZE - 1);
            e.repo[REPO_SIZE - 1] = '\0';
            std::strncpy(e.filename, filename.c_str(), FILENAME_SIZE - 1);
            e.filename[FILENAME_SIZE - 1] = '\0';
            std::strncpy(e.downloadURL, downloadURL.c_str(), URL_SIZE - 1);
            e.downloadURL[URL_SIZE - 1] = '\0';
            e.active = true;
            count++;
            return true;
        }
    }
    
    return false;
}

std::optional<std::string> HFModelMapper::getRepo(const std::string& key) const {
    for (const auto& e : entries) {
        if (e.active && std::strcmp(e.key, key.c_str()) == 0) {
            return std::string(e.repo);
        }
    }
    return std::nullopt;
}

std::optional<std::string> HFModelMapper::getFilename(const std::string& key) const {
    for (const auto& e : entries) {
        if (e.active && std::strcmp(e.key, key.c_str()) == 0) {
            return std::string(e.filename);
        }
    }
    return std::nullopt;
}

std::optional<std::string> HFModelMapper::getDownloadURL(const std::string& key) const {
    for (const auto& e : entries) {
        if (e.active && std::strcmp(e.key, key.c_str()) == 0) {
            return std::string(e.downloadURL);
        }
    }
    return std::nullopt;
}

bool HFModelMapper::remove(const std::string& key) {
    for (auto& e : entries) {
        if (e.active && std::strcmp(e.key, key.c_str()) == 0) {
            e.active = false;
            count--;
            return true;
        }
    }
    return false;
}

std::vector<std::string> HFModelMapper::keys() const {
    std::vector<std::string> result;
    for (const auto& e : entries) {
        if (e.active) {
            result.push_back(std::string(e.key));
        }
    }
    return result;
}

void HFModelMapper::clear() {
    for (auto& e : entries) {
        e.active = false;
    }
    count = 0;
}

bool HFModelMapper::save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;
    
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));
    file.write(reinterpret_cast<const char*>(entries.data()), 
               sizeof(Entry) * MAX_ENTRIES);
    
    return file.good();
}

bool HFModelMapper::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;
    
    file.read(reinterpret_cast<char*>(&count), sizeof(count));
    file.read(reinterpret_cast<char*>(entries.data()), 
              sizeof(Entry) * MAX_ENTRIES);
    
    return file.good();
}

bool HFModelMapper::exportToText(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) return false;
    
    for (const auto& e : entries) {
        if (e.active) {
            file << e.key << "|" << e.repo << "|" << e.filename << "|" << e.downloadURL << "\n";
        }
    }
    
    return file.good();
}

bool HFModelMapper::importFromText(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) return false;
    
    clear();
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        // Parse: key|repo|filename|downloadURL
        std::vector<std::string> parts;
        size_t start = 0;
        size_t pos;
        
        while ((pos = line.find('|', start)) != std::string::npos) {
            parts.push_back(line.substr(start, pos - start));
            start = pos + 1;
        }
        parts.push_back(line.substr(start));
        
        if (parts.size() >= 4) {
            // Trim whitespace from each part
            for (auto& part : parts) {
                part.erase(0, part.find_first_not_of(" \t"));
                part.erase(part.find_last_not_of(" \t") + 1);
            }
            
            if (!set(parts[0], parts[1], parts[2], parts[3])) {
                return false;
            }
        }
    }
    
    return true;
}
