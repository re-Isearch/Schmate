#include <fstream>
#include <string>
#include <iostream>
#include <sys/sysinfo.h>
#include <unistd.h>

// Method 1: Parse /proc/meminfo (your current approach)
size_t get_mem_available_procfs() {
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    
    while (std::getline(meminfo, line)) {
        if (line.substr(0, 12) == "MemAvailable") {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                size_t kb = std::stoull(line.substr(pos + 1));
                return kb * 1024;  // Convert KB to bytes
            }
        }
    }
    return 0;
}

// Method 2: Use sysinfo() system call (simpler, faster)
size_t get_mem_available_sysinfo() {
    struct sysinfo info;
    if (sysinfo(&info) != 0) {
        return 0;
    }
    
    // Available memory = free + buffers + cached (approximately)
    // But sysinfo doesn't give us the exact "MemAvailable" metric
    return info.freeram * info.mem_unit;
}

// Method 3: More accurate sysinfo approach
struct MemoryInfo {
    size_t total;
    size_t free;
    size_t available;  // Best estimate
    size_t buffers;
    size_t cached;
    size_t used;
};

MemoryInfo get_memory_info_sysinfo() {
    struct sysinfo info;
    MemoryInfo mem = {};
    
    if (sysinfo(&info) != 0) {
        return mem;
    }
    
    mem.total = info.totalram * info.mem_unit;
    mem.free = info.freeram * info.mem_unit;
    mem.buffers = info.bufferram * info.mem_unit;
    // Note: sysinfo doesn't provide cached separately, 
    // so we estimate available as free + buffers
    mem.available = mem.free + mem.buffers;
    mem.used = mem.total - mem.free - mem.buffers;
    
    return mem;
}

// Method 4: Full /proc/meminfo parser (most accurate)
MemoryInfo get_memory_info_procfs() {
    MemoryInfo mem = {};
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    
    auto parse_kb = [](const std::string& line) -> size_t {
        size_t pos = line.find(':');
        if (pos != std::string::npos) {
            return std::stoull(line.substr(pos + 1)) * 1024;  // KB to bytes
        }
        return 0;
    };
    
    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal:") == 0) {
            mem.total = parse_kb(line);
        } else if (line.find("MemFree:") == 0) {
            mem.free = parse_kb(line);
        } else if (line.find("MemAvailable:") == 0) {
            mem.available = parse_kb(line);
        } else if (line.find("Buffers:") == 0) {
            mem.buffers = parse_kb(line);
        } else if (line.find("Cached:") == 0) {
            mem.cached = parse_kb(line);
        }
    }
    
    mem.used = mem.total - mem.available;
    return mem;
}

// Method 5: Using getrusage for process-specific memory
#include <sys/resource.h>

size_t get_process_memory_usage() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        // ru_maxrss is in KB on Linux, bytes on macOS
        #ifdef __linux__
        return usage.ru_maxrss * 1024;
        #else
        return usage.ru_maxrss;
        #endif
    }
    return 0;
}

// Method 6: Reading /proc/self/status (process memory)
struct ProcessMemoryInfo {
    size_t vm_size;    // Virtual memory size
    size_t vm_rss;     // Resident set size (physical memory)
    size_t vm_data;    // Data segment size
};

ProcessMemoryInfo get_process_memory_status() {
    ProcessMemoryInfo mem = {};
    std::ifstream status("/proc/self/status");
    std::string line;
    
    auto parse_kb = [](const std::string& line) -> size_t {
        size_t pos = line.find(':');
        if (pos != std::string::npos) {
            return std::stoull(line.substr(pos + 1)) * 1024;
        }
        return 0;
    };
    
    while (std::getline(status, line)) {
        if (line.find("VmSize:") == 0) {
            mem.vm_size = parse_kb(line);
        } else if (line.find("VmRSS:") == 0) {
            mem.vm_rss = parse_kb(line);
        } else if (line.find("VmData:") == 0) {
            mem.vm_data = parse_kb(line);
        }
    }
    
    return mem;
}

// Comparison and recommendations
void compare_methods() {
    std::cout << "=== System Memory (All Methods) ===\n";
    
    // Method 1: Your original
    size_t avail1 = get_mem_available_procfs();
    std::cout << "MemAvailable (procfs): " << (avail1 / 1024 / 1024) << " MB\n";
    
    // Method 2: sysinfo
    size_t avail2 = get_mem_available_sysinfo();
    std::cout << "Free (sysinfo):        " << (avail2 / 1024 / 1024) << " MB\n";
    
    // Method 3: Full sysinfo
    auto info3 = get_memory_info_sysinfo();
    std::cout << "\nsysinfo() details:\n";
    std::cout << "  Total:     " << (info3.total / 1024 / 1024) << " MB\n";
    std::cout << "  Free:      " << (info3.free / 1024 / 1024) << " MB\n";
    std::cout << "  Available: " << (info3.available / 1024 / 1024) << " MB\n";
    std::cout << "  Used:      " << (info3.used / 1024 / 1024) << " MB\n";
    
    // Method 4: Full procfs
    auto info4 = get_memory_info_procfs();
    std::cout << "\n/proc/meminfo details:\n";
    std::cout << "  Total:     " << (info4.total / 1024 / 1024) << " MB\n";
    std::cout << "  Free:      " << (info4.free / 1024 / 1024) << " MB\n";
    std::cout << "  Available: " << (info4.available / 1024 / 1024) << " MB\n";
    std::cout << "  Buffers:   " << (info4.buffers / 1024 / 1024) << " MB\n";
    std::cout << "  Cached:    " << (info4.cached / 1024 / 1024) << " MB\n";
    std::cout << "  Used:      " << (info4.used / 1024 / 1024) << " MB\n";
    
    // Method 5: Process memory
    std::cout << "\n=== Process Memory ===\n";
    size_t proc_mem = get_process_memory_usage();
    std::cout << "Process RSS (getrusage): " << (proc_mem / 1024 / 1024) << " MB\n";
    
    // Method 6: Process details
    auto proc_info = get_process_memory_status();
    std::cout << "Process memory (/proc/self/status):\n";
    std::cout << "  VmSize: " << (proc_info.vm_size / 1024 / 1024) << " MB\n";
    std::cout << "  VmRSS:  " << (proc_info.vm_rss / 1024 / 1024) << " MB\n";
    std::cout << "  VmData: " << (proc_info.vm_data / 1024 / 1024) << " MB\n";
}

int main() {
    compare_methods();
    
    std::cout << "\n=== KEY DIFFERENCES ===\n";
    std::cout << "MemAvailable: Memory that can be allocated WITHOUT swapping\n";
    std::cout << "              Includes reclaimable caches and buffers\n";
    std::cout << "              BEST for determining if allocation will succeed\n\n";
    
    std::cout << "MemFree:      Completely unused memory\n";
    std::cout << "              Lower than MemAvailable (doesn't include caches)\n";
    std::cout << "              Too conservative for allocation decisions\n\n";
    
    std::cout << "sysinfo():    Fast system call, less accurate\n";
    std::cout << "              Gives 'free' but not true 'available'\n";
    std::cout << "              Good for quick checks\n\n";
    
    std::cout << "/proc/meminfo: Most accurate, includes MemAvailable\n";
    std::cout << "               Slightly slower (file I/O)\n";
    std::cout << "               Best for allocation planning\n\n";
    
    std::cout << "=== RECOMMENDATION ===\n";
    std::cout << "For cache sizing: Use MemAvailable from /proc/meminfo (your current method)\n";
    std::cout << "For quick checks: Use sysinfo() if speed matters\n";
    std::cout << "For process monitoring: Use /proc/self/status or getrusage()\n";
    
    return 0;
}

/*
 * WHY MemAvailable vs Other Metrics:
 * 
 * 1. MemAvailable (kernel 3.14+):
 *    - BEST metric for "can I allocate X bytes?"
 *    - Includes memory that can be reclaimed (page cache, buffers)
 *    - Linux kernel's estimate of available memory
 *    - Example: 2GB free + 4GB cache = ~6GB available
 * 
 * 2. MemFree:
 *    - Only completely unused memory
 *    - Too conservative (ignores reclaimable memory)
 *    - Example: 2GB free, but 6GB truly available
 * 
 * 3. sysinfo() freeram:
 *    - Similar to MemFree (doesn't include reclaimable)
 *    - Faster (syscall vs file I/O)
 *    - Less accurate for allocation decisions
 * 
 * 4. For your cache use case:
 *    - Stick with MemAvailable from /proc/meminfo ✓
 *    - It's the most accurate for sizing allocations
 *    - Small performance cost is worth the accuracy
 * 
 * 5. When to use sysinfo():
 *    - Frequent polling (every second)
 *    - Don't need exact available memory
 *    - Want minimal overhead
 * 
 * 6. Optimization:
 *    - Cache the file handle if calling frequently
 *    - Use mmap + parse if doing many reads
 *    - For single checks, your method is fine
 */
