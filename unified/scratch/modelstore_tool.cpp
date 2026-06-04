#include "ModelPathStore.hpp"
#include <iostream>
#include <iomanip>

// g++ -std=c++17 -O3 ModelPathStore.cpp HFModelMapper.cpp modelstore_tool.cpp -o modelstore

void printUsage(const char* prog) {
    std::cout << "Usage:\n"
              << "  " << prog << " load <text_file> <db_file>     # Import text to database\n"
              << "  " << prog << " dump <db_file> <text_file>     # Export database to text\n"
              << "  " << prog << " set <db_file> <key> <value>    # Set a key-value pair\n"
              << "  " << prog << " get <db_file> <key>            # Get value for key\n"
              << "  " << prog << " remove <db_file> <key>         # Remove a key\n"
              << "  " << prog << " list <db_file>                 # List all entries\n"
              << "  " << prog << " clear <db_file>                # Clear all entries\n"
              << "\nText file format: key=value (one per line, # for comments)\n";
}

void listEntries(ModelPathStore& store) {
    size_t count = 0;
    std::cout << std::left << std::setw(20) << "Key" << " | " << "Value\n";
    std::cout << std::string(80, '-') << "\n";
    
    for (size_t i = 0; i < ModelPathStore::MAX_ENTRIES; i++) {
        auto& entry = store.entries[i];
        if (entry.active) {
            std::cout << std::setw(20) << entry.key << " | " << entry.value << "\n";
            count++;
        }
    }
    std::cout << "\nTotal entries: " << count << "/" << ModelPathStore::MAX_ENTRIES << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string cmd = argv[1];
    ModelPathStore store;
    
    if (cmd == "load" && argc == 4) {
        std::string textFile = argv[2];
        std::string dbFile = argv[3];
        
        if (!store.importFromText(textFile)) {
            std::cerr << "Error: Failed to import from " << textFile << "\n";
            return 1;
        }
        
        if (!store.save(dbFile)) {
            std::cerr << "Error: Failed to save to " << dbFile << "\n";
            return 1;
        }
        
        std::cout << "Successfully loaded " << store.size() << " entries from " 
                  << textFile << " to " << dbFile << "\n";
        return 0;
    }
    
    if (cmd == "dump" && argc == 4) {
        std::string dbFile = argv[2];
        std::string textFile = argv[3];
        
        if (!store.load(dbFile)) {
            std::cerr << "Error: Failed to load from " << dbFile << "\n";
            return 1;
        }
        
        if (!store.exportToText(textFile)) {
            std::cerr << "Error: Failed to export to " << textFile << "\n";
            return 1;
        }
        
        std::cout << "Successfully dumped " << store.size() << " entries from " 
                  << dbFile << " to " << textFile << "\n";
        return 0;
    }
    
    if (cmd == "set" && argc == 5) {
        std::string dbFile = argv[2];
        std::string key = argv[3];
        std::string value = argv[4];
        
        store.load(dbFile); // Load existing, ignore if doesn't exist
        
        if (!store.set(key, value)) {
            std::cerr << "Error: Failed to set key (check size limits or capacity)\n";
            return 1;
        }
        
        if (!store.save(dbFile)) {
            std::cerr << "Error: Failed to save to " << dbFile << "\n";
            return 1;
        }
        
        std::cout << "Set: " << key << " = " << value << "\n";
        return 0;
    }
    
    if (cmd == "get" && argc == 4) {
        std::string dbFile = argv[2];
        std::string key = argv[3];
        
        if (!store.load(dbFile)) {
            std::cerr << "Error: Failed to load from " << dbFile << "\n";
            return 1;
        }
        
        auto value = store.get(key);
        if (value) {
            std::cout << *value << "\n";
            return 0;
        } else {
            std::cerr << "Key not found: " << key << "\n";
            return 1;
        }
    }
    
    if (cmd == "remove" && argc == 4) {
        std::string dbFile = argv[2];
        std::string key = argv[3];
        
        if (!store.load(dbFile)) {
            std::cerr << "Error: Failed to load from " << dbFile << "\n";
            return 1;
        }
        
        if (!store.remove(key)) {
            std::cerr << "Key not found: " << key << "\n";
            return 1;
        }
        
        if (!store.save(dbFile)) {
            std::cerr << "Error: Failed to save to " << dbFile << "\n";
            return 1;
        }
        
        std::cout << "Removed: " << key << "\n";
        return 0;
    }
    
    if (cmd == "list" && argc == 3) {
        std::string dbFile = argv[2];
        
        if (!store.load(dbFile)) {
            std::cerr << "Error: Failed to load from " << dbFile << "\n";
            return 1;
        }
        
        listEntries(store);
        return 0;
    }
    
    if (cmd == "clear" && argc == 3) {
        std::string dbFile = argv[2];
        
        store.clear();
        
        if (!store.save(dbFile)) {
            std::cerr << "Error: Failed to save to " << dbFile << "\n";
            return 1;
        }
        
        std::cout << "Cleared all entries from " << dbFile << "\n";
        return 0;
    }
    
    printUsage(argv[0]);
    return 1;
}
