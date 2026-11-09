cmake_minimum_required(VERSION 3.16)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(SchmateQ8Regression LANGUAGES CXX)
    enable_testing()
endif()

find_package(Threads REQUIRED)
add_executable(test_q8_passthrough ../test_q8_passthrough.cpp)
target_compile_features(test_q8_passthrough PRIVATE cxx_std_20)
target_include_directories(test_q8_passthrough PRIVATE ../../include)
target_link_libraries(test_q8_passthrough PRIVATE Threads::Threads)
if(TARGET sbert_core)
    target_link_libraries(test_q8_passthrough PRIVATE sbert_core)
else()
    # Exercise the production search and persistence code without a model or
    # inference backend, using the same focused sources as run_energy_tests.sh.
    target_sources(test_q8_passthrough PRIVATE
        ../../src/unified_hnsw.cpp ../../src/HnswConfig.cpp
        ../../src/Logger.cpp ../../src/FileLock.cpp)
    target_compile_options(test_q8_passthrough PRIVATE -ffunction-sections -fdata-sections)
    if(APPLE)
        target_link_options(test_q8_passthrough PRIVATE -Wl,-dead_strip)
    else()
        target_link_options(test_q8_passthrough PRIVATE -Wl,--gc-sections)
    endif()
endif()
add_test(NAME q8_passthrough COMMAND test_q8_passthrough)
