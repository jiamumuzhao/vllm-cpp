cmake_minimum_required(VERSION 3.22)
project(vllm-cpp VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA compute capability (RTX 2080 Ti = 75)")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math -allow-unsupported-compiler")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CompilerSettings)
include(Dependencies)
find_package(CUDAToolkit REQUIRED)

# ── Library ─────────────────────────────────────────────────
add_library(vllm STATIC)
add_library(vllm::vllm ALIAS vllm)

target_include_directories(vllm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

target_link_libraries(vllm PUBLIC CUDA::cublas CUDA::cublasLt CUDA::cudart)

if(VLLM_NETWORK_DEPS)
    target_compile_definitions(vllm PUBLIC VLLM_USE_SPDLOG)
    target_link_libraries(vllm PUBLIC spdlog::spdlog nlohmann_json::nlohmann_json)
endif()

if(NOT TARGET httplib)
    add_library(httplib INTERFACE)
endif()

# ── Sources ─────────────────────────────────────────────────
file(GLOB_RECURSE VLLM_SOURCES "src/*.cpp" "src/*.cu")
target_sources(vllm PRIVATE ${VLLM_SOURCES})

# ── Executable ──────────────────────────────────────────────
add_executable(vllm_server src/main.cpp)
target_link_libraries(vllm_server PRIVATE vllm::vllm httplib)
set_target_properties(vllm_server PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# ── Tests ───────────────────────────────────────────────────
option(VLLM_ENABLE_TESTS "Build tests" ON)
if(VLLM_ENABLE_TESTS AND VLLM_NETWORK_DEPS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── Benchmarks ──────────────────────────────────────────────
option(VLLM_ENABLE_BENCHMARKS "Build benchmarks" ON)
if(VLLM_ENABLE_BENCHMARKS AND VLLM_NETWORK_DEPS)
    add_subdirectory(benchmarks)
endif()
