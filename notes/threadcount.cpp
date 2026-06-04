#include <iostream>
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

int main() {
    int threadCount = getThreadCount();
    
    std::cout << "Number of available hardware threads: " << threadCount << std::endl;
    
    return 0;
}
