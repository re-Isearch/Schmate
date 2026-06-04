cmake_minimum_required(VERSION 3.16)
project(sbert_search)

# =========================
# C++ Standard
# =========================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================
# Options
# =========================
option(USE_THREADPOOL "Use cached thread pool for parallel_search" OFF)


# ====================================================================
# Enforce Binary Realignment and Hardware Acceleration Macros
# ====================================================================
if(APPLE)
    # Force BOTH libraries to see the exact same struct fields
    add_compile_definitions(GGML_USE_METAL=1)
    add_compile_definitions(GGML_USE_ACCELERATE=1)
    
    # Ensure optimization matching prevents memory reordering bugs
    add_compile_options(-O3 -fPIC)
endif()

# =========================
# External Subdirectories (Modernized!)
# =========================
# This automatically registers named targets: 'ggml' and 'bert' (or whatever bert.cpp defines)
#add_subdirectory(ggml)

# ====================================================================
# Peer-Directory Dependency Integration (Symlink Isolation)
# ====================================================================

# Force CMake to treat the symlink path exactly as a local subdirectory 
# rather than escaping out to the absolute realpath sibling directory.
add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/bert.cpp"    # The logical path inside Schmate
    "${CMAKE_CURRENT_BINARY_DIR}/bert_build"  # Isolate all its compile files here!
)

# =========================
# Core sources (library only)
# =========================
set(CORE_SOURCES
    src/BertIndex.cpp
    src/HnswConfig.cpp
    src/ShardedIndex.cpp
    src/BertIndexManager.cpp
    src/Util.cpp
    src/OffsetFile.cpp
    src/Logger.cpp
    src/FileLock.cpp
    src/SBertGGML.cpp
    src/hf_mapper.cpp
    src/StringEmbedding.cpp
    src/FileStreamCache.cpp
    src/unified_hnsw.cpp
)

set(EXTENDED_SOURCES
    ${CORE_SOURCES}
    src/EmbeddingIndexer.cpp
)

# =========================
# STATIC LIB
# =========================
add_library(sbert_core STATIC ${CORE_SOURCES})

target_include_directories(sbert_core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include        # Crucial: Maps absolutely to /Volumes/Celeritas/CoreQuarry/Schmate/include
        ${CMAKE_CURRENT_SOURCE_DIR}/bert.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/bert.cpp/ggml
)

target_link_libraries(sbert_core
    PUBLIC
        bert   # <-- Link directly to the named target from add_subdirectory
        ggml   # <-- Link directly to the named target from add_subdirectory
)

# =========================
# SHARED LIB
# =========================
add_library(sbert_shared SHARED ${CORE_SOURCES})


# ─────────────────────────────────────────────
# Resolve the true physical path of the ib symlink
# ─────────────────────────────────────────────
get_filename_component(IB_PHYSICAL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/ib/src" REALPATH)

# Diagnostic verification print
message(STATUS "[Schmate Config] Resolved ib/src realpath to: ${IB_PHYSICAL_SRC}")


target_include_directories(sbert_shared
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include        # engine headers
        ${CMAKE_CURRENT_SOURCE_DIR}/bert.cpp       # Pulls bert.h cleanly
        ${CMAKE_CURRENT_SOURCE_DIR}/bert.cpp/ggml  # Pulls matching ggml.h cleanly
)

target_link_libraries(sbert_shared
    PUBLIC
        bert
        ggml
)

set_target_properties(sbert_shared PROPERTIES
    OUTPUT_NAME "sbert"
)

# =========================
# Threadpool
# =========================
if(USE_THREADPOOL)
    target_compile_definitions(sbert_core PUBLIC USE_THREADPOOL)
    target_compile_definitions(sbert_shared PUBLIC USE_THREADPOOL)
endif()

# =========================
# OpenMP
# =========================
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sbert_core PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(sbert_shared PUBLIC OpenMP::OpenMP_CXX)
endif()

# =========================
# Executables
# =========================
add_executable(sbert_search src/main.cpp)
add_executable(config_editor utils/config_editor.cpp)
add_executable(test_reconstruct tests/test_reconstruct.cpp)

# =========================
# Link executables → shared lib
# =========================
target_link_libraries(sbert_search PRIVATE sbert_shared)
target_link_libraries(config_editor PRIVATE sbert_shared)
target_link_libraries(test_reconstruct PRIVATE sbert_shared)

# =========================
# Optimization
# =========================
#target_compile_options(sbert_core PRIVATE -O3)
#target_compile_options(sbert_shared PRIVATE -O3)

# =========================
# macOS specific fixes
# =========================
if(APPLE)
    # Enable rpath globally
    set(CMAKE_MACOSX_RPATH ON)

    # Shared lib install name
    set_target_properties(sbert_shared PROPERTIES
        INSTALL_NAME_DIR "@rpath"
	VERSION 2.0
	SOVERSION 2
    )

    # Crucial Fix: Let your executables discover both your shared lib 
    # AND the newly compiled nested GGML dylibs at runtime.
    set_target_properties(sbert_search config_editor test_reconstruct PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        BUILD_RPATH "@loader_path"
        INSTALL_RPATH "@executable_path"
    )

    # Accelerate framework (GGML handles this internally now, but kept if your core source needs it)
    target_link_libraries(sbert_core PUBLIC "-framework Accelerate")
    target_link_libraries(sbert_shared PUBLIC "-framework Accelerate")
endif()

target_link_options(sbert_shared PRIVATE
    "-Wl,-undefined,dynamic_lookup"
)

# =========================
# EXTENDED LIB
# =========================
add_library(sbert_extended SHARED ${EXTENDED_SOURCES})

set_target_properties(sbert_extended PROPERTIES
    VERSION 2.0
    SOVERSION 2
)

# Apply to sbert_extended (or whichever target compiles BertIndex.cpp)
target_include_directories(sbert_extended
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include        # Crucial: Maps absolutely to /Volumes/Celeritas/CoreQuarry/Schmate/include
        ${CMAKE_CURRENT_SOURCE_DIR}/bert.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/bert.cpp/ggml
        ${CMAKE_SOURCE_DIR}/include
        ${IB_PHYSICAL_SRC}                         # True physical path to common.hxx
        lib/src

)

target_link_libraries(sbert_extended
    PUBLIC
        bert
        ggml
)

target_link_options(sbert_extended PRIVATE
    "-Wl,-undefined,dynamic_lookup"
)

set_target_properties(sbert_extended PROPERTIES
    OUTPUT_NAME "schmate"
)



# ==========================================
# macOS Runtime Dependency Paths (RPATH Fixing)
# ==========================================
if(APPLE)
    # 1. Force RPATH on globally
    set(CMAKE_MACOSX_RPATH ON)

    # 2. Tell the shared libraries to register themselves as relocatable
    set_target_properties(sbert_shared sbert_extended PROPERTIES
        INSTALL_NAME_DIR "@rpath"
    )

    # 3. Hard-code search zones inside your CLI Executables
    # This instructs dyld to search exactly where the executable runs, 
    # as well as inside the nested build artifact output directories.
    set_target_properties(sbert_search config_editor test_reconstruct PROPERTIES
        BUILD_WITH_INSTALL_RPATH FALSE
        BUILD_RPATH "@loader_path;@loader_path/bert.cpp;@loader_path/bert.cpp/ggml/src"
        INSTALL_RPATH "@executable_path;@executable_path/bert.cpp;@executable_path/bert.cpp/ggml/src"
    )

    # 4. Accelerate framework binding
    target_link_libraries(sbert_core PUBLIC "-framework Accelerate")
    target_link_libraries(sbert_shared PUBLIC "-framework Accelerate")
endif()
