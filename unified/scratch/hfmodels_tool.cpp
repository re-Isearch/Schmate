#include "HFModelMapper.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>

// clang++ -O3 HFModelMapper.cpp hfmodels_tool.cpp -o hfmodels
// g++ -std=c++17 -O3 HFModelMapper.cpp hfmodels_tool.cpp -o hfmodels

void printUsage(const char* prog) {
    std::cout << "Usage:\n"
              << "  " << prog << " load <text_file> <db_file>          # Import text to database\n"
              << "  " << prog << " dump <db_file> <text_file>          # Export database to text\n"
              << "  " << prog << " set <db_file> <key> <repo> <file>   # Set a mapping\n"
              << "  " << prog << " get <db_file> <key>                 # Get mapping info\n"
              << "  " << prog << " url <db_file> <key>                 # Get download URL\n"
              << "  " << prog << " remove <db_file> <key>              # Remove a mapping\n"
              << "  " << prog << " list <db_file>                      # List all mappings\n"
              << "  " << prog << " clear <db_file>                     # Clear all mappings\n"
              << "\nText file format: key=repo|filename (one per line, # for comments)\n"
              << "Example: llama2=TheBloke/Llama-2-7B-GGUF|llama-2-7b.Q4_K_M.gguf\n";
}

void listMappings(const HFModelMapper& mapper) {
    std::cout << std::left << std::setw(15) << "Key" << " | " 
              << std::setw(35) << "Repository" << " | " << "Filename\n";
    std::cout << std::string(100, '-') << "\n";
    
    // Get and sort keys for consistent output
    auto keyList = mapper.keys();
    std::sort(keyList.begin(), keyList.end());
    
    for (const auto& key : keyList) {
        auto repo = mapper.getRepo(key);
        auto filename = mapper.getFilename(key);
        
        if (repo && filename) {
            std::cout << std::setw(15) << key << " | " 
                      << std::setw(35) << *repo << " | " 
                      << *filename << "\n";
        }
    }
    
    std::cout << "\nTotal mappings: " << mapper.size() << "/" << HFModelMapper::MAX_ENTRIES << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string cmd = argv[1];
    HFModelMapper mapper;
    
    if (cmd == "load" && argc == 4) {
        std::string textFile = argv[2];
        std::string dbFile = argv[3];
        
        if (!mapper.importFromText(textFile)) {
            std::cerr << "Error: Failed to import from " << textFile << "\n";
            return 1;
        }
        
        if (!mapper.save(dbFile)) {
            std::cerr << "Error: Failed to save to " << dbFile << "\n";
            return 1;
        }
        
        std::cout << "Successfully loaded " << mapper.size() << " mappings from " 
                  << textFile << " to " << dbFile << "\n";
        return 0;
    }
    
    if (cmd == "dump" && argc == 4) {
        std::string dbFile = argv[2];
        std::string textFile = argv[3];
        
        if (!mapper.load(dbFile)) {
            std::cerr << "Error: Failed to load from " << dbFile << "\n";
            return 1;
        }
        
        if (!mapper.exportToText(textFile)) {
            std::cerr << "Error: Failed to export to " << textFile << "\n";
            return 1;
        }
        
        std::cout << "Successfully dumped " << mapper.size() << " mappings from " 
                  << dbFile << " to " << textFile << "\n";
        return 0;
    }
    
    if (cmd == "set" && argc == 6) {
        std::string dbFile = argv[2];
        std::string key = argv[3];
        std::string repo = argv[4];
        std::string filename = argv[5];
        
        mapper.load(dbFile); // Load existing, ignore if doesn't exist
        
        if (!mapper.set(key, repo, filename)) {
            std::cerr << "Error: Failed to set mapping (capacity reached or invalid input)\n";
            return 1;
        }
        
        if (!mapper.save(dbFile)) {
            std::cerr << "Error: Failed to save to " << dbFile << "\n";
            return 1;
        }
        
        std::cout << "Set mapping: " << key << " -> " << repo << "/" << filename << "\n";
        return 0;
    }
    
    if (cmd == "get" && argc == 4) {
        std::string dbFile = argv[2];
        std::string key = argv[3];
        
        if (!mapper.load(dbFile)) {
            std::cerr << "Error: Failed to load from " << dbFile << "\n";
            return 1;
        }
        
        auto repo = mapper.getRepo(key);
        auto filename = mapper.getFilename(key);
        
        if (repo && filename) {
            std::cout << "Key:      " << key << "\n";
            std::cout << "Repo:     " << *repo << "\n";
            std::cout << "File:     " << *filename << "\n";
            std::cout << "Path:     " << *repo << "/" << *filename << "\n";
            return 0;
        } else {
            std::cerr << "Key not found: " << key << "\n";
            return 1;
        }
    }
    
    if (cmd == "url" && argc == 4) {
        std::string dbFile = argv[2];
        std::string key = argv[3];
        
        if (!mapper.load(dbFile)) {
            std::cerr << "Error: Failed to load from " << dbFile << "\n";
            return 1;
        }
        
        auto url = mapper.getDownloadURL(key);
        if (url) {
            std::cout << *url << "\n";
            return 0;
        } else {
            std::cerr << "Key not found: " << key << "\n";
            return 1;
        }
    }
    
    if (cmd == "remove" && argc == 4) {
        std::string dbFile = argv[2];
        std::string key = argv[3];
        
        if (!mapper.load(dbFile)) {
            std::cerr << "Error: Failed to load from " << dbFile << "\n";
            return 1;
        }
        
        if (!mapper.remove(key)) {
            std::cerr << "Key not found: " << key << "\n";
            return 1;
        }
        
        if (!mapper.save(dbFile)) {
            std::cerr << "Error: Failed to save to " << dbFile << "\n";
            return 1;
        }
        
        std::cout << "Removed mapping: " << key << "\n";
        return 0;
    }
    
    if (cmd == "list" && argc == 3) {
        std::string dbFile = argv[2];
        
        if (!mapper.load(dbFile)) {
            std::cerr << "Error: Failed to load from " << dbFile << "\n";
            return 1;
        }
        
        listMappings(mapper);
        return 0;
    }
    
    if (cmd == "clear" && argc == 3) {
        std::string dbFile = argv[2];
        
        mapper.clear();
        
        if (!mapper.save(dbFile)) {
            std::cerr << "Error: Failed to save to " << dbFile << "\n";
            return 1;
        }
        
        std::cout << "Cleared all mappings from " << dbFile << "\n";
        return 0;
    }
    
    printUsage(argv[0]);
    return 1;
}
