#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <iostream>

class SparseMmapCache {
private:
    int fd;
    void* map;
    size_t size;
    std::string filepath;

public:
    SparseMmapCache(const std::string& path, size_t cache_size) 
        : fd(-1), map(MAP_FAILED), size(cache_size), filepath(path) 
    {
        // Open or create file
        fd = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
        if (fd == -1) {
            throw std::runtime_error("Failed to open cache file");
        }

        // Get current file size
        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            ::close(fd);
            throw std::runtime_error("Failed to stat cache file");
        }

        // If file is smaller than desired size, extend it
        // Using ftruncate creates a SPARSE file (no actual disk allocation yet)
        if ((size_t)sb.st_size < cache_size) {
            if (ftruncate(fd, cache_size) == -1) {
                ::close(fd);
                throw std::runtime_error("Failed to extend cache file");
            }
            std::cout << "Created sparse file: " << path 
                      << " (logical size: " << cache_size << " bytes)\n";
        }

        // Memory map the file
        map = ::mmap(nullptr, cache_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) {
            ::close(fd);
            throw std::runtime_error("Failed to mmap cache file");
        }

        std::cout << "Mapped cache: " << cache_size << " bytes at " << map << "\n";
    }

    ~SparseMmapCache() {
        if (map != MAP_FAILED) {
            // Sync changes to disk before unmapping
            msync(map, size, MS_SYNC);
            munmap(map, size);
        }
        if (fd != -1) {
            close(fd);
        }
    }

    // Write data at offset (this allocates disk blocks on-demand)
    void write(size_t offset, const void* data, size_t len) {
        if (offset + len > size) {
            throw std::out_of_range("Write beyond cache bounds");
        }
        
        // Writing to sparse regions automatically allocates blocks
        std::memcpy((char*)map + offset, data, len);
    }

    // Read data at offset (reads zeros from unwritten sparse regions)
    void read(size_t offset, void* data, size_t len) const {
        if (offset + len > size) {
            throw std::out_of_range("Read beyond cache bounds");
        }
        
        // Reading from unwritten sparse regions returns zeros
        std::memcpy(data, (char*)map + offset, len);
    }

    // Direct pointer access (use with care)
    void* get_ptr(size_t offset = 0) {
        if (offset >= size) {
            throw std::out_of_range("Offset beyond cache bounds");
        }
        return (char*)map + offset;
    }

    // Get actual disk usage (how much is really allocated)
    size_t get_actual_size() const {
        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            return 0;
        }
        // st_blocks is in 512-byte blocks
        return sb.st_blocks * 512;
    }

    // Get logical size (full sparse size)
    size_t get_logical_size() const {
        return size;
    }

    // Sync to disk (optional, happens automatically on close)
    void sync() {
        if (map != MAP_FAILED) {
            msync(map, size, MS_SYNC);
        }
    }

    // Advise kernel about access patterns
    void advise_sequential() {
        madvise(map, size, MADV_SEQUENTIAL);
    }

    void advise_random() {
        madvise(map, size, MADV_RANDOM);
    }

    void advise_willneed(size_t offset, size_t len) {
        if (offset + len <= size) {
            madvise((char*)map + offset, len, MADV_WILLNEED);
        }
    }
};

// Example: Cache with header and entries
struct CacheHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t entry_count;
    uint64_t next_offset;
};

struct CacheEntry {
    uint64_t key;
    uint64_t offset;
    uint32_t size;
    uint32_t flags;
};

class StructuredCache {
private:
    SparseMmapCache cache;
    static constexpr size_t HEADER_SIZE = sizeof(CacheHeader);
    static constexpr size_t ENTRY_SIZE = sizeof(CacheEntry);

public:
    StructuredCache(const std::string& path, size_t max_size)
        : cache(path, max_size)
    {
        // Initialize header if new
        CacheHeader* header = (CacheHeader*)cache.get_ptr(0);
        if (header->magic != 0xCAFEBABE) {
            header->magic = 0xCAFEBABE;
            header->version = 1;
            header->entry_count = 0;
            header->next_offset = HEADER_SIZE;
            cache.sync();
        }
    }

    void add_entry(uint64_t key, const void* data, size_t data_size) {
        CacheHeader* header = (CacheHeader*)cache.get_ptr(0);
        
        // Check if we have space
        size_t needed = ENTRY_SIZE + data_size;
        if (header->next_offset + needed > cache.get_logical_size()) {
            throw std::runtime_error("Cache full");
        }

        // Write entry metadata
        CacheEntry entry;
        entry.key = key;
        entry.offset = header->next_offset + ENTRY_SIZE;
        entry.size = data_size;
        entry.flags = 0;
        
        size_t entry_pos = HEADER_SIZE + header->entry_count * ENTRY_SIZE;
        cache.write(entry_pos, &entry, ENTRY_SIZE);

        // Write actual data (sparse blocks allocated here)
        cache.write(entry.offset, data, data_size);

        // Update header
        header->entry_count++;
        header->next_offset += needed;
    }

    bool get_entry(uint64_t key, void* data, size_t max_size) {
        CacheHeader* header = (CacheHeader*)cache.get_ptr(0);
        
        // Linear search (you'd use a hash table in production)
        for (uint64_t i = 0; i < header->entry_count; i++) {
            size_t entry_pos = HEADER_SIZE + i * ENTRY_SIZE;
            CacheEntry entry;
            cache.read(entry_pos, &entry, ENTRY_SIZE);
            
            if (entry.key == key) {
                if (entry.size > max_size) {
                    throw std::runtime_error("Buffer too small");
                }
                cache.read(entry.offset, data, entry.size);
                return true;
            }
        }
        return false;
    }

    void print_stats() const {
        std::cout << "Cache stats:\n";
        std::cout << "  Logical size: " << cache.get_logical_size() << " bytes\n";
        std::cout << "  Actual disk usage: " << cache.get_actual_size() << " bytes\n";
        std::cout << "  Compression ratio: " 
                  << (float)cache.get_logical_size() / cache.get_actual_size() 
                  << "x\n";
    }
};

// Example usage
int main() {
    try {
        // Create a 1GB sparse cache (won't use 1GB of disk immediately)
        std::cout << "=== Basic Sparse Cache ===\n";
        SparseMmapCache cache("test_cache.dat", 1024 * 1024 * 1024);
        
        std::cout << "Initial state:\n";
        std::cout << "  Logical size: " << cache.get_logical_size() << " bytes\n";
        std::cout << "  Actual size: " << cache.get_actual_size() << " bytes\n";
        
        // Write some data (this allocates blocks on-demand)
        const char* data1 = "Hello, sparse world!";
        cache.write(0, data1, strlen(data1) + 1);
        
        const char* data2 = "More data at offset 1MB";
        cache.write(1024 * 1024, data2, strlen(data2) + 1);
        
        std::cout << "\nAfter writing:\n";
        std::cout << "  Actual size: " << cache.get_actual_size() << " bytes\n";
        std::cout << "  (Only allocated blocks for written data)\n";
        
        // Read back
        char buffer[100];
        cache.read(0, buffer, 50);
        std::cout << "Read from offset 0: " << buffer << "\n";
        
        cache.read(1024 * 1024, buffer, 50);
        std::cout << "Read from offset 1MB: " << buffer << "\n";
        
        // Read from unwritten region (returns zeros)
        cache.read(2048 * 1024, buffer, 50);
        std::cout << "Read from unwritten region: first byte = " 
                  << (int)buffer[0] << " (should be 0)\n";
        
        // Structured cache example
        std::cout << "\n=== Structured Cache ===\n";
        StructuredCache scache("struct_cache.dat", 100 * 1024 * 1024);
        
        // Add some entries
        const char* value1 = "Cached value for key 42";
        scache.add_entry(42, value1, strlen(value1) + 1);
        
        const char* value2 = "Another cached value";
        scache.add_entry(123, value2, strlen(value2) + 1);
        
        // Retrieve
        char result[100];
        if (scache.get_entry(42, result, sizeof(result))) {
            std::cout << "Found key 42: " << result << "\n";
        }
        
        scache.print_stats();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}

/*
 * KEY POINTS ABOUT SPARSE FILES:
 * 
 * 1. Creation:
 *    - ftruncate() creates sparse file (instant, no disk I/O)
 *    - File shows as 1GB but uses only a few KB on disk
 * 
 * 2. Block allocation:
 *    - Disk blocks allocated ONLY when you write to them
 *    - Automatic, transparent to your code
 *    - No explicit "allocate block" call needed
 * 
 * 3. Reading unwritten regions:
 *    - Returns zeros (no actual disk read)
 *    - Fast and safe
 * 
 * 4. Performance:
 *    - Perfect for caches that fill gradually
 *    - No upfront allocation delay
 *    - Writes allocate blocks on-demand (small overhead)
 * 
 * 5. Limitations:
 *    - Can fail mid-operation if disk fills up
 *    - Some backup tools don't handle sparse files well
 *    - NFS may have issues (depends on version)
 * 
 * 6. Best practices:
 *    - Check disk space before large operations
 *    - Use msync(MS_SYNC) for critical data
 *    - Consider madvise() for access patterns
 *    - Monitor actual disk usage with get_actual_size()
 * 
 * 7. Filesystem support:
 *    - ext4, xfs, btrfs: full support ✓
 *    - NTFS: supported ✓
 *    - FAT32: NO sparse file support ✗
 */
