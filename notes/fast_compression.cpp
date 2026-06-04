#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>

// LZ4 is the fastest - install: apt-get install liblz4-dev (Linux) or brew install lz4 (macOS)
#include <lz4.h>

// Zstd is also excellent - install: apt-get install libzstd-dev (Linux) or brew install zstd (macOS)
#include <zstd.h>

class FastCompressor {
public:
    enum Algorithm {
        LZ4_ALGO,
        ZSTD_ALGO
    };
    
    // LZ4 compression - FASTEST option
    static std::vector<char> compress_lz4(const char* data, size_t size) {
        int max_compressed_size = LZ4_compressBound(size);
        std::vector<char> compressed(max_compressed_size + 4); // +4 for size header
        
        // Store original size in first 4 bytes
        *reinterpret_cast<uint32_t*>(compressed.data()) = static_cast<uint32_t>(size);
        
        int compressed_size = LZ4_compress_default(
            data,
            compressed.data() + 4,
            size,
            max_compressed_size
        );
        
        if (compressed_size <= 0) {
            throw std::runtime_error("LZ4 compression failed");
        }
        
        compressed.resize(compressed_size + 4);
        return compressed;
    }
    
    static std::vector<char> decompress_lz4(const char* compressed_data, size_t compressed_size) {
        if (compressed_size < 4) {
            throw std::runtime_error("Invalid compressed data");
        }
        
        uint32_t original_size = *reinterpret_cast<const uint32_t*>(compressed_data);
        std::vector<char> decompressed(original_size);
        
        int result = LZ4_decompress_safe(
            compressed_data + 4,
            decompressed.data(),
            compressed_size - 4,
            original_size
        );
        
        if (result < 0) {
            throw std::runtime_error("LZ4 decompression failed");
        }
        
        return decompressed;
    }
    
    // Zstd compression - Slightly slower but better compression
    static std::vector<char> compress_zstd(const char* data, size_t size, int level = 1) {
        size_t max_compressed_size = ZSTD_compressBound(size);
        std::vector<char> compressed(max_compressed_size);
        
        // Use level 1 for maximum speed, up to 3 for balanced speed/compression
        size_t compressed_size = ZSTD_compress(
            compressed.data(),
            max_compressed_size,
            data,
            size,
            level  // 1 = fastest, 3 = good balance, 22 = maximum compression
        );
        
        if (ZSTD_isError(compressed_size)) {
            throw std::runtime_error(std::string("Zstd compression failed: ") + 
                                   ZSTD_getErrorName(compressed_size));
        }
        
        compressed.resize(compressed_size);
        return compressed;
    }
    
    static std::vector<char> decompress_zstd(const char* compressed_data, size_t compressed_size) {
        unsigned long long original_size = ZSTD_getFrameContentSize(compressed_data, compressed_size);
        
        if (original_size == ZSTD_CONTENTSIZE_ERROR || 
            original_size == ZSTD_CONTENTSIZE_UNKNOWN) {
            throw std::runtime_error("Invalid Zstd compressed data");
        }
        
        std::vector<char> decompressed(original_size);
        
        size_t result = ZSTD_decompress(
            decompressed.data(),
            original_size,
            compressed_data,
            compressed_size
        );
        
        if (ZSTD_isError(result)) {
            throw std::runtime_error(std::string("Zstd decompression failed: ") + 
                                   ZSTD_getErrorName(result));
        }
        
        return decompressed;
    }
};

// Benchmark helper
class CompressionBenchmark {
public:
    static void benchmark(const std::string& text) {
        std::cout << "Original size: " << text.size() << " bytes\n";
        std::cout << "Text preview: " << text.substr(0, 50) << "...\n\n";
        
        // LZ4 benchmark
        {
            auto start = std::chrono::high_resolution_clock::now();
            auto compressed = FastCompressor::compress_lz4(text.c_str(), text.size());
            auto compress_end = std::chrono::high_resolution_clock::now();
            auto decompressed = FastCompressor::decompress_lz4(compressed.data(), compressed.size());
            auto decompress_end = std::chrono::high_resolution_clock::now();
            
            auto compress_time = std::chrono::duration<double, std::micro>(compress_end - start).count();
            auto decompress_time = std::chrono::duration<double, std::micro>(decompress_end - compress_end).count();
            
            std::cout << "LZ4:\n";
            std::cout << "  Compressed size: " << compressed.size() << " bytes ("
                      << (100.0 * compressed.size() / text.size()) << "%)\n";
            std::cout << "  Compression time: " << compress_time << " μs\n";
            std::cout << "  Decompression time: " << decompress_time << " μs\n";
            std::cout << "  Throughput: " << (text.size() / compress_time) << " MB/s\n\n";
        }
        
        // Zstd level 1 benchmark
        {
            auto start = std::chrono::high_resolution_clock::now();
            auto compressed = FastCompressor::compress_zstd(text.c_str(), text.size(), 1);
            auto compress_end = std::chrono::high_resolution_clock::now();
            auto decompressed = FastCompressor::decompress_zstd(compressed.data(), compressed.size());
            auto decompress_end = std::chrono::high_resolution_clock::now();
            
            auto compress_time = std::chrono::duration<double, std::micro>(compress_end - start).count();
            auto decompress_time = std::chrono::duration<double, std::micro>(decompress_end - compress_end).count();
            
            std::cout << "Zstd (level 1):\n";
            std::cout << "  Compressed size: " << compressed.size() << " bytes ("
                      << (100.0 * compressed.size() / text.size()) << "%)\n";
            std::cout << "  Compression time: " << compress_time << " μs\n";
            std::cout << "  Decompression time: " << decompress_time << " μs\n";
            std::cout << "  Throughput: " << (text.size() / compress_time) << " MB/s\n\n";
        }
        
        // Zstd level 3 benchmark
        {
            auto start = std::chrono::high_resolution_clock::now();
            auto compressed = FastCompressor::compress_zstd(text.c_str(), text.size(), 3);
            auto compress_end = std::chrono::high_resolution_clock::now();
            auto decompressed = FastCompressor::decompress_zstd(compressed.data(), compressed.size());
            auto decompress_end = std::chrono::high_resolution_clock::now();
            
            auto compress_time = std::chrono::duration<double, std::micro>(compress_end - start).count();
            auto decompress_time = std::chrono::duration<double, std::micro>(decompress_end - compress_end).count();
            
            std::cout << "Zstd (level 3):\n";
            std::cout << "  Compressed size: " << compressed.size() << " bytes ("
                      << (100.0 * compressed.size() / text.size()) << "%)\n";
            std::cout << "  Compression time: " << compress_time << " μs\n";
            std::cout << "  Decompression time: " << decompress_time << " μs\n";
            std::cout << "  Throughput: " << (text.size() / compress_time) << " MB/s\n";
        }
    }
};

int main() {
    // Example 1: Short text (typical sentence)
    std::string short_text = "The quick brown fox jumps over the lazy dog. "
                            "This is a test of fast compression algorithms for short texts.";
    
    std::cout << "=== SHORT TEXT BENCHMARK ===\n";
    CompressionBenchmark::benchmark(short_text);
    
    std::cout << "\n=== MEDIUM TEXT BENCHMARK ===\n";
    // Example 2: Medium text (paragraph)
    std::string medium_text = 
        "In computer science, data compression is the process of encoding information "
        "using fewer bits than the original representation. Compression can be either "
        "lossy or lossless. Lossless compression reduces bits by identifying and eliminating "
        "statistical redundancy. No information is lost in lossless compression. Lossy compression "
        "reduces bits by removing unnecessary or less important information. The trade-off between "
        "compression speed and compression ratio is an important consideration.";
    
    CompressionBenchmark::benchmark(medium_text);
    
    return 0;
}

/*
Compilation instructions:

Linux:
  sudo apt-get install liblz4-dev libzstd-dev
  g++ -std=c++11 -O3 fast_compression.cpp -o fast_compression -llz4 -lzstd

macOS:
  brew install lz4 zstd

 g++ -std=c++11 -O3 "-I$(brew --prefix lz4)/include" "-I$(brew --prefix zstd)/include" "-L$(brew --prefix lz4)/lib"   "-L$(brew --prefix zstd)/lib" fast_compression.cpp -o fast_compression -llz4 -lzstd

Windows (MSVC with vcpkg):
  vcpkg install lz4 zstd
  cl /EHsc /O2 fast_compression.cpp lz4.lib zstd.lib

RECOMMENDATION:
- Use LZ4 for absolute maximum speed (3000+ MB/s compression)
- Use Zstd level 1-3 for better compression with still very fast speed (1000+ MB/s)
- Both decompress extremely fast (5000+ MB/s)
- For texts under 512 bytes, LZ4 is probably the best bet
*/
