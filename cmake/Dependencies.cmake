option(VLLM_NETWORK_DEPS "Download third-party dependencies" ON)

if(NOT VLLM_NETWORK_DEPS)
    message(STATUS "CUDA-only build mode")
    add_library(httplib INTERFACE)
    return()
endif()

include(FetchContent)

FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.14.1
)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(cpp_httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.18.3
)
FetchContent_MakeAvailable(cpp_httplib)

if(NOT TARGET httplib)
    add_library(httplib ALIAS cpp_httplib)
endif()

if(VLLM_ENABLE_TESTS)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
        OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt ON" "BUILD_GMOCK OFF"
    )
    FetchContent_MakeAvailable(googletest)
endif()

if(VLLM_ENABLE_BENCHMARKS)
    FetchContent_Declare(benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.1
        OPTIONS "BENCHMARK_ENABLE_TESTING OFF"
    )
    FetchContent_MakeAvailable(benchmark)
endif()

message(STATUS "Dependencies resolved via FetchContent")
