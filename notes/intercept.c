// intercept.c - compile as shared lib and LD_PRELOAD it
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>
#include <execinfo.h>

static int (*real_close)(int)       = NULL;
static int (*real_fclose)(FILE*)    = NULL;

__attribute__((constructor))
static void init() {
    real_close  = dlsym(RTLD_NEXT, "close");
    real_fclose = dlsym(RTLD_NEXT, "fclose");
}

int close(int fd) {
    if (fd > 3) {   // ignore stdin/out/err
        void* bt[16];
        int n = backtrace(bt, 16);
        fprintf(stderr, "close(fd=%d)\n", fd);
        backtrace_symbols_fd(bt, n, STDERR_FILENO);
    }
    return real_close(fd);
}

int fclose(FILE* f) {
    if (f) {
        void* bt[16];
        int n = backtrace(bt, 16);
        fprintf(stderr, "fclose(fd=%d)\n", fileno(f));
        backtrace_symbols_fd(bt, n, STDERR_FILENO);
    }
    return real_fclose(f);
}
