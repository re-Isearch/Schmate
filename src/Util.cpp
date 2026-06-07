#include "Util.hpp"
#include "IO.hpp"
#include "Logger.hpp"
#include "hf_mapper.hpp"


#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <string>

#include <string>
#include <unordered_map>
#include <array>

#include <pwd.h>

using namespace hnswlib;

/*

GGML Codes:

| Code (`f16` value) | Quantization Type | Bits per weight | Description                     |
| :----------------: | :---------------- | :-------------: | :------------------------------ |
|        **0**       | `F32`             |        32       | Full precision                  |
|        **1**       | `F16`             |        16       | Half precision                  |
|        **2**       | `Q4_0`            |        4        | 4-bit block quantization        |
|        **3**       | `Q4_1`            |        4        | 4-bit with scale + bias         |
|      Beyond here bert.cpp does not provide support (4&5 are not even defined)              |
|        **6**       | `Q5_0`            |        5        | 5-bit quantization              |
|        **7**       | `Q5_1`            |        5        | 5-bit w/ bias                   |
|        **8**       | `Q8_0`            |        8        | 8-bit uniform                   |
|        **9**       | `Q8_1`            |        8        | 8-bit w/ bias                   |
|       **10**       | `Q2_K`            |        2        | 2-bit “K-block” quantization    |
|       **11**       | `Q3_K`            |        3        | 3-bit K-block                   |
|       **12**       | `Q4_K`            |        4        | 4-bit K-block base              |
|       **13**       | `Q5_K`            |        5        | 5-bit K-block                   |
|       **14**       | `Q6_K`            |        6        | 6-bit K-block                   |
|       **15**       | `Q8_K`            |        8        | 8-bit K-block                   |
|       **16**       | `IQ2_XXS`         |        2        | “Improved Quantization” variant |
|       **17**       | `IQ2_XS`          |        2        |                                 |
|       **18**       | `IQ3_XS`          |        3        |                                 |
|       **19**       | `IQ1_S`           |        1        | 1-bit ultra-low precision       |

In some newer implmentations we see some different values > 19. 

*/


// The metadata in GGUF uses a **different** enum from the enum defined
// in the ggml tensor library backend!
// (and yes this wasted a lot of time to figure out)
//
// We convert the type numbers from GGUF here into canonical names
// for the ggml backend that we can convert back into ggml codes.
static inline std::string gguf_file_type_name(uint32_t ft) {
    switch (ft) {
        case 0:  return "F32";
        case 1:  return "F16";
        case 2:  return "Q4_0";
        case 3:  return "Q4_1";
        case 7:  return "Q8_0";
        case 8:  return "Q5_0";
        case 9:  return "Q5_1";
        case 10: return "Q2_K";
        
        // Fold Q3 sub-types down to the core engine layout
        case 11: // Q3_K_S
        case 12: // Q3_K_M
        case 13: // Q3_K_L
            return "Q3_K";
            
        // Fold Q4 sub-types down (This maps straight to your 4-bit pipeline!)
        case 14: // Q4_K_S
        case 15: // Q4_K_M
            return "Q4_K";
            
        // Fold Q5 sub-types down
        case 16: // Q5_K_S
        case 17: // Q5_K_M
            return "Q5_K";
            
        case 18: return "Q6_K";
        
        // I-Quant (Importance Matrix) Mappings for extreme low-bit setups
        case 19: return "IQ2_XXS";
        case 20: return "IQ2_XS";
        case 21: return "IQ3_XXS";
        case 22: return "IQ1_S";
        case 23: return "IQ4_NL";
        case 24: return "IQ3_S";
        case 25: return "IQ2_S";
        case 26: return "IQ4_XS";
        
        default: return "UNKNOWN(" + std::to_string(ft) + ")";
    }
}


namespace {
    struct QuantMapping {
        uint32_t             code;         // 32-bit int code in the .gmml file
        const char*          name;         // Quantization Type name (.gguf)
        hnswlib::StorageType storage_type; // Int storage (important for pass-through)
    };
    

// The major issue are code 4 and 20. In some older models 20 was BF16. In newer that 
// code is a 4-bit quantization type and BF16 is code 30. 4 was by Ollama a FP4 now 39
//
// So depending upon version. 20 can be 4-bit int or a 16-bit floating point!
// A 4 can be an obsolete 4-bit int or a new 4-bit floating point.

    // See ggml/include/ggml.h
    constexpr std::array<QuantMapping, 34> quant_mappings = {{
        {0, "F32", StorageType::FLOAT32},
        {1, "F16", StorageType::FP16},
        {2, "Q4_0", StorageType::INT4},
        {3, "Q4_1", StorageType::INT4},

        // bert.cpp stops here
        // Rest are from ggml.h
//      {4, "Q4_2", StorageType::INT4}, // Support has been removed!
//      {5, "Q4_3", StorageType::INT4}, // Support has been removed!

        {6, "Q5_0", StorageType::INT5},
        {7, "Q5_1", StorageType::INT5},
        {8, "Q8_0", StorageType::INT8},
        {9, "Q8_1", StorageType::INT8},
        {10, "Q2_K", StorageType::INT2},
        {11, "Q3_K", StorageType::INT3},
        {12, "Q4_K", StorageType::INT4},
        {13, "Q5_K", StorageType::INT5},
        {14, "Q6_K", StorageType::INT6},
        {15, "Q8_K", StorageType::INT8},
        {16, "IQ2_XXS", StorageType::INT2},
        {17, "IQ2_XS", StorageType::INT2},
        {18, "IQ3_XXS", StorageType::INT3},
        {19, "IQ1_S", StorageType::BIN1},
        {20, "IQ4_NL", StorageType::INT4},  // Note: original code had "BF16" here
        {21, "IQ3_S", StorageType::INT3},
        {22, "IQ2_S", StorageType::INT2},
        {23, "IQ4_XS", StorageType::INT4},
        {24, "I8", StorageType::INT8},
        {25, "I16", StorageType::INT16},
        {26, "I32", StorageType::INT32},
        {27, "I64", StorageType::INT64}, // GGML does not support 64 bits
        {28, "F64", StorageType::FLOAT64}, // GGML does not support Fp64
        {29, "IQ1_M", StorageType::BIN1},
        {30, "BF16", StorageType::FP16},  // BFloat16 support added here, waas 20
        // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
        // GGML_TYPE_Q4_0_4_8 = 32,
        // GGML_TYPE_Q4_0_8_8 = 33,
        {34, "TQ1_0", StorageType::BIN1},
        {35, "TQ2_0", StorageType::INT2},
        // GGML_TYPE_IQ4_NL_4_4 = 36,
        // GGML_TYPE_IQ4_NL_4_8 = 37,
        // GGML_TYPE_IQ4_NL_8_8 = 38,

        // Below are very new
        {39, "MXFP4", StorageType::FP4},
        {40, "NVFP4", StorageType::FP4}, //  Nvidia specific E4M3 with block scaling

        {41, "Q1_0", StorageType::BIN1}  // Extremely low-bit quantization
/*
        Name: "MXFP4" or sometimes "mxfp4-e8m0"
        Microscaling 4-bit Floating Point with 8-bit exponent, 0-bit mantissa

        A native training format: Unlike other quantization methods that reduce precision
        after training, gpt-oss models were trained directly in MXFP4 format Code number 39
        in the GGML type enum 4-bit format: Uses 4 bits per weight but maintains higher
        quality than traditional Q4 quantization
        Block-based: Stores weights in blocks of 32 elements with shared scaling factors

        NOTE: There was an inconsistency where Ollama initially used code 4 for MXFP4, but
        llama.cpp officially assigned it code 39. This caused compatibility issues with models
        converted using different tools.
*/



    }};
}

const std::string ggml_quant_name(uint32_t code) {
    for (const auto& mapping : quant_mappings) {
        if (mapping.code == code) {
            return mapping.name;
        }
    }
    return "Unknown(" + std::to_string(code) + ")";
}

int ggml_name_to_quant(const std::string& name) {
    static std::unordered_map<std::string, int> name_map;
    
    // Build map on first call
    if (name_map.empty()) {
        for (const auto& mapping : quant_mappings) {
            name_map[mapping.name] = mapping.code;
        }
    }
    
    auto it = name_map.find(name);
    return (it != name_map.end()) ? it->second : -1;
}



hnswlib::StorageType ggml_quant_type(uint32_t code) {
    for (const auto& mapping : quant_mappings) {
        if (mapping.code == code) {
            return mapping.storage_type;
        }
    }
    return hnswlib::StorageType::FLOAT32;  // Default for unknown codes
}


// The below could have also used the above.. 
hnswlib::StorageType ggml_name_to_quant_type(const std::string& name) {
    static std::unordered_map<std::string, StorageType> type_map;
    
    // Build map on first call
    if (type_map.empty()) {
        for (const auto& mapping : quant_mappings) {
            type_map[mapping.name] = mapping.storage_type;
        }
    }
    
    auto it = type_map.find(name);
    return (it != type_map.end()) ? it->second : StorageType::FLOAT32;
}


///////////


std::optional<GmmlHparams> read_ggml_info(const std::string& path) {
   GmmlHparams info;
   // File exists and is large enough
   if ( file_size(path) > (off_t)sizeof(GmmlHparams)) {
       std::ifstream fin(path, std::ios::binary);
       if (fin) {
           return read_le<GmmlHparams>(fin);
//         GmmlHparams info;
//         fin.read(reinterpret_cast<char*>(&info), sizeof(info));
//         return info;
        }
    }
   return std::nullopt;
}

std::pair<hnswlib::StorageType, std::string>
	get_ggml_model_quant(const std::string &filepath)
{
    auto hdr = read_ggml_info (filepath);
    int code = hdr ? hdr->f16 : 0;
    return  {  ggml_quant_type(code), ggml_quant_name(code) };
}


static GGML_TYPE FileType(const std::string& filepath) {
    enum GGML_TYPE type = GGML_TYPE::UNKNOWN;
    // Check for valid GGML magic numbers
    const uint32_t GGML_MAGIC = 0x67676d6c; // "ggml"
    const uint32_t GGJT_MAGIC = 0x67676a74; // "ggjt"
    const uint32_t GGLA_MAGIC = 0x67676c61; // "ggla"
    const uint32_t GGMF_MAGIC = 0x67676d66; // "ggmf"
    const uint32_t GGUF_MAGIC = 0x46554747; // "GGUF" (Stored as 'G' 'G' 'U' 'F')

    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        return type;
    }
    // Read exactly 4 bytes into our magic variable
    uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    magic = __builtin_bswap32(magic);
#endif
     bool valid_magic = (magic == GGML_MAGIC || magic == GGJT_MAGIC || 
                        magic == GGLA_MAGIC || magic == GGMF_MAGIC ||
                        magic == GGUF_MAGIC);
     if (!valid_magic) {
	// std::cerr << "Invalid magic number: 0x" << std::hex << magic << std::dec << std::endl;
        return type ;
     }
   type = ( magic == GGUF_MAGIC) ? GGML_TYPE::GGUF : GGML_TYPE::GGML;
   return type ;
}


enum GGUFMetadataValueType : uint32_t {
    GGUF_MVT_UINT8   = 0,
    GGUF_MVT_INT8    = 1,
    GGUF_MVT_UINT16  = 2,
    GGUF_MVT_INT16   = 3,
    GGUF_MVT_UINT32  = 4,
    GGUF_MVT_INT32   = 5,
    GGUF_MVT_FLOAT32 = 6,
    GGUF_MVT_BOOL    = 7,
    GGUF_MVT_STRING  = 8,
    GGUF_MVT_ARRAY   = 9,
    GGUF_MVT_UINT64  = 10,
    GGUF_MVT_INT64   = 11,
    GGUF_MVT_FLOAT64 = 12,
};

// Helper utility to read Little-Endian values safely
//template<typename T>
//T read_le(std::ifstream &f) {
//    T val;
//    f.read(reinterpret_cast<char*>(&val), sizeof(T));
//    return val;
//}

// GGUF strings use an explicit uint64_t length prefix
std::string read_gguf_string(std::ifstream &f) {
    uint64_t len = read_le<uint64_t>(f);
    if (len > 65536) return "";
    std::string str(len, '\0');
    f.read(&str[0], len);
    if (f.fail()) return "";
    return str;
}


std::optional<GGUFInfo> read_gguf_info(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::nullopt;

    // 1. Verify Magic Number ("GGUF" in ASCII format)
    uint32_t magic = read_le<uint32_t>(f);
    if (magic != 0x46554747) { 
        return std::nullopt; 
    }

    uint32_t version = read_le<uint32_t>(f);
    uint64_t n_tensors = read_le<uint64_t>(f); // Tensor count ALWAYS comes first!
    uint64_t n_kv = read_le<uint64_t>(f);      // KV dictionary item count ALWAYS comes second!

    GGUFInfo info;

    // 2. Safely Process Key-Value Metadata without losing cursor tracking
    for (uint64_t i = 0; i < n_kv; i++) {
        if (!f.good()) break;
        
        std::string key = read_gguf_string(f);
        uint32_t vtype = read_le<uint32_t>(f);

        if (vtype == GGUF_MVT_STRING) {
            std::string value = read_gguf_string(f);
            if (key == "general.architecture") {
                info.architecture = value;
            }
	   // Text-based fallback mapping for pooling if an exporter used raw text
	   else if (key ==  "sentence_embedding.pooling_name" ||
		key == "sentence_embedding.pooling_type" || 
		key == "embedding.pooling_type" || 
		key == "pooling_type" ||
		key == "sentence_embedding.architecture" ||
		key == "pooling_name") {
                if (value == "mean" || value == "MEAN" || value == "1")      info.pooling_type = 1;
                else if (value == "cls" || value == "CLS" || value == "2")   info.pooling_type = 2;
                else if (value == "none" || value == "NONE" || value == "0") info.pooling_type = 0;
            }
        }
        else if (vtype == GGUF_MVT_UINT32 || vtype == GGUF_MVT_INT32) {
            uint32_t v = read_le<uint32_t>(f);
            if (key == "llama.embedding_length" || key == "bert.embedding_length" || key == "embedding_length") {
                info.embedding_length = v;
            } else if (key == "general.file_type") {
	       info.quant_type = gguf_file_type_name(v);
	       info.f16 =  ggml_name_to_quant(info.quant_type);
	    } else if (key == "sentence_embedding.pooling_type" || 
                key == "embedding.pooling_type" || 
		key == "bert.pooling_type" ||
                key == "pooling_type") {
                info.pooling_type = static_cast<int32_t>(v);
            }
        }
        else if (vtype == GGUF_MVT_UINT64 || vtype == GGUF_MVT_INT64) {
            uint64_t v = read_le<uint64_t>(f);
            if (key == "llama.embedding_length" || key == "bert.embedding_length" || key == "embedding_length") {
                info.embedding_length = static_cast<uint32_t>(v);
            }
	    // Added regardless of width 
            else if (key == "sentence_embedding.pooling_type" || 
                     key == "embedding.pooling_type" || 
                     key == "pooling_type") {
                info.pooling_type = static_cast<int32_t>(v);
            }
        }
        else if (vtype == GGUF_MVT_ARRAY) {
            uint32_t arr_type = read_le<uint32_t>(f);
            uint64_t arr_len  = read_le<uint64_t>(f);

            if (arr_type == GGUF_MVT_STRING) {
                for (uint64_t j = 0; j < arr_len; ++j) {
                    uint64_t s_len = read_le<uint64_t>(f);
                    f.seekg(s_len, std::ios::cur);
                }
            } else {
                uint64_t element_size = 0;
                switch (arr_type) {
                    case GGUF_MVT_UINT8:   case GGUF_MVT_INT8:  case GGUF_MVT_BOOL: element_size = 1; break;
                    case GGUF_MVT_UINT16:  case GGUF_MVT_INT16:                     element_size = 2; break;
                    case GGUF_MVT_UINT32:  case GGUF_MVT_INT32: case GGUF_MVT_FLOAT32: element_size = 4; break;
                    case GGUF_MVT_UINT64:  case GGUF_MVT_INT64: case GGUF_MVT_FLOAT64: element_size = 8; break;
                    default: break;
                }
                if (element_size > 0) {
                    f.seekg(element_size * arr_len, std::ios::cur);
                }
            }
        } else {
            // Precise fall-through sizes for primitive items
            uint64_t skip_size = 0;
            switch (vtype) {
                case GGUF_MVT_UINT8:   case GGUF_MVT_INT8:  case GGUF_MVT_BOOL: skip_size = 1; break;
                case GGUF_MVT_UINT16:  case GGUF_MVT_INT16:                     skip_size = 2; break;
                case GGUF_MVT_UINT32:  case GGUF_MVT_INT32: case GGUF_MVT_FLOAT32: skip_size = 4; break;
                case GGUF_MVT_UINT64:  case GGUF_MVT_INT64: case GGUF_MVT_FLOAT64: skip_size = 8; break;
                default: break;
            }
            if (skip_size > 0) {
                f.seekg(skip_size, std::ios::cur);
            }
        }
    }

    info.is_bert_cpp_compatible = (info.architecture == "bert" || info.architecture == "nomic-bert");
    // If it's a Bert-style model but has no pooling info, bert.cpp might struggle 
    // to know which layer to extract.
    if (info.is_bert_cpp_compatible && info.pooling_type == -1) {
      LOG_WARN_S() << "Model may be incompatible with bert.cpp\n";
    }

    return info;
}



#include <filesystem>
#include <optional>
#include <sstream>

namespace fs = std::filesystem;

std::vector<std::string> splitPath(const std::string& pathStr, char delimiter)
{
    std::vector<std::string> paths;
    std::stringstream ss(pathStr);
    std::string path;
    
    while (std::getline(ss, path, delimiter)) {
        if (!path.empty()) {
            paths.push_back(path);
        }
    }
    
    return paths;
}

std::pair<std::string, GGML_TYPE> find_model(const std::string& identity, const std::string& searchPaths) {
  static auto map = ModelMap();
  std::string filename = identity;
  if (map) {
     auto result = map-> getModel ( identity );
     if (result)
	filename = result->second;
   }
   return find_ggml_model(filename, searchPaths);
}


// GGML model finder
std::pair<std::string, GGML_TYPE>
find_ggml_model(const std::string& filename, const std::string& searchPaths) {
    
    auto result = findInPathsWithData<GGML_TYPE>(
        filename, 
        searchPaths,
        [](const fs::path& path) -> std::optional<GGML_TYPE> {
            auto type = FileType(path.string());
            if (type != GGML_TYPE::UNKNOWN) {
                return type;
            }
            return std::nullopt;
        });
    
    if (result) {
        return {result->first.string(), result->second};
    }
    
    return {filename, GGML_TYPE::UNKNOWN};
}

#ifdef __APPLE__
#include <malloc/malloc.h>
#include <dlfcn.h>
#include <iostream>

// Optional safety shim in case we build on older SDKs.
typedef void (*malloc_zone_pressure_relief_t)(void*, size_t);

void relax_macos_malloc_zones() {
    // On macOS 11+, malloc_zone_pressure_relief() is available.
    void* handle = dlopen("/usr/lib/libSystem.dylib", RTLD_NOW);
    if (!handle) return;

    malloc_zone_pressure_relief_t fn =
        (malloc_zone_pressure_relief_t)dlsym(handle, "malloc_zone_pressure_relief");

    if (fn) {
        malloc_zone_t* default_zone = malloc_default_zone();
        // 0 means "release as much as you can"
        fn(default_zone, 0);
        // std::cerr << "[INFO] macOS allocator zones relaxed.\n";
    }
    dlclose(handle);
}
#endif

#include <thread>

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
#elif defined(__APPLE__) || defined(__MACH__)
    #include <sys/sysctl.h>
#elif defined(__linux__) || defined(__unix__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)
    #include <unistd.h>
    #if defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)
        #include <sys/sysctl.h>
    #endif
#endif

int getThreadCount() {
    // Try the standard C++ method first
    unsigned int threadCount = std::thread::hardware_concurrency();
    
    if (threadCount != 0) {
        return threadCount;
    }
    
    // Fallback to platform-specific methods if hardware_concurrency() returns 0
    #if defined(_WIN32) || defined(_WIN64)
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        return sysinfo.dwNumberOfProcessors;
    
    #elif defined(__APPLE__) || defined(__MACH__)
        int nm[2];
        size_t len = 4;
        uint32_t count;
        
        nm[0] = CTL_HW;
        nm[1] = HW_AVAILCPU;
        sysctl(nm, 2, &count, &len, NULL, 0);
        
        if(count < 1) {
            nm[1] = HW_NCPU;
            sysctl(nm, 2, &count, &len, NULL, 0);
            if(count < 1) { count = 1; }
        }
        return count;
    
    #elif defined(__linux__)
        return sysconf(_SC_NPROCESSORS_ONLN);
    
    #elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)
        int nm[2];
        size_t len = 4;
        int count;
        
        nm[0] = CTL_HW;
        nm[1] = HW_NCPU;
        sysctl(nm, 2, &count, &len, NULL, 0);
        
        return count;
    
    #else
        return 1; // Default fallback
    #endif
}



std::pair<hnswlib::StorageType, std::string>
        get_gguf_model_quant(const std::string &filepath)
{
    auto hdr = read_gguf_info (filepath); 
    if (hdr) return  { ggml_quant_type(hdr->f16), hdr->quant_type };
    return { hnswlib::StorageType::FLOAT32, ggml_quant_name(-1) };
}    


std::pair<hnswlib::StorageType, std::string>
        get_model_quant(const std::string &filepath)
{       
   switch (FileType(filepath)) {
      case GGML_TYPE::GGUF :
	return get_gguf_model_quant(filepath);
      case GGML_TYPE::GGML :
        return get_ggml_model_quant(filepath);
      default:
        return  { hnswlib::StorageType::FLOAT32, ggml_quant_name(-1) };
    }
}
    

inline std::string expandTilde(std::string path)
{
    if (path.empty() || path[0] != '~')
        return path;

    auto slash = path.find('/');

    // ~ or ~/...
    if (slash == 1 || slash == std::string::npos)
    {
        const char *home = getenv("HOME");

        if (!home)
        {
            if (auto *pw = getpwuid(getuid()))
                home = pw->pw_dir;
        }

        if (!home)
            return path;

        if (slash == std::string::npos)
            return home;

        return std::string(home) + path.substr(1);
    }

    // ~user or ~user/...
    std::string user = path.substr(1, slash - 1);

    if (auto *pw = getpwnam(user.c_str()))
    {
        if (slash == std::string::npos)
            return pw->pw_dir;

        return std::string(pw->pw_dir) + path.substr(slash);
    }

    return path;
}


#include <regex>

inline std::string expandEnvironment(std::string path)
{
    static const std::regex re(
        R"(\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*))");

    std::string result;
    std::smatch match;

    while (std::regex_search(path, match, re))
    {
        result += match.prefix();

        std::string var =
            match[1].matched ? match[1].str() : match[2].str();

        if (const char *value = getenv(var.c_str()))
            result += value;

        path = match.suffix();
    }

    result += path;

    return result;
}

std::filesystem::path resolvePath(std::string_view input)
{
    std::string path(input);

    // Expand ~ and ~user
    path = expandTilde(path);

    // Expand $VAR and ${VAR}
    path = expandEnvironment(path);

    std::error_code ec;

    auto p = std::filesystem::path(path);

    if (std::filesystem::exists(p, ec))
        return std::filesystem::weakly_canonical(p, ec);

    return p.lexically_normal();
}



