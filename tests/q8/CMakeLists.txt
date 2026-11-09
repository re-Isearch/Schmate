cmake_minimum_required(VERSION 3.16)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(SchmateQ8Regression LANGUAGES CXX)
    enable_testing()
endif()

find_package(Threads REQUIRED)
add_executable(test_q8_passthrough ../test_q8_passthrough.cpp)
target_compile_features(test_q8_passthrough PRIVATE cxx_std_17)
target_include_directories(test_q8_passthrough PRIVATE ../../include)
# The regression exercises Q8's scalar packing/metric path; avoid unrelated
# optional floating-point SIMD dispatch when compiling this standalone target.
target_compile_definitions(test_q8_passthrough PRIVATE NO_MANUAL_VECTORIZATION)
target_link_libraries(test_q8_passthrough PRIVATE Threads::Threads)
add_test(NAME q8_passthrough COMMAND test_q8_passthrough)
