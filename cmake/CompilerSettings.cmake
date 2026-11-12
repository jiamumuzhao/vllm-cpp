# CompilerSettings.cmake
# C++17 strict compilation settings, warnings as errors, CUDA arch validation

include(CheckCXXCompilerFlag)

# ── C++ compiler flags ──────────────────────────────────────
add_compile_options(
  -Wall
  -Wextra

  -Wshadow
  -Wnon-virtual-dtor

  -Woverloaded-virtual


  -Wnull-dereference

  -Wformat=2
  -Wimplicit-fallthrough
)

# GCC 15 specific: suppress noisy-but-harmless warnings
check_cxx_compiler_flag(-Wno-psabi HAS_WNO_PSABI)
if(HAS_WNO_PSABI)
  add_compile_options(-Wno-psabi)
endif()

# Release build optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3 -DNDEBUG -funroll-loops -finline-functions)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-march=native -mtune=native)
  endif()
endif()

# Debug build: sanitizers
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(-g -O0 -fno-omit-frame-pointer)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    check_cxx_compiler_flag(-fsanitize=address HAS_ASAN)
    check_cxx_compiler_flag(-fsanitize=undefined HAS_UBSAN)
    if(HAS_ASAN AND HAS_UBSAN)
      add_compile_options(-fsanitize=address -fsanitize=undefined)
      add_link_options(-fsanitize=address -fsanitize=undefined)
    endif()
  endif()
endif()

# CUDA-specific flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math")

# The default targets the current development GPU (RTX 2080 Ti, sm_75).
# Override with -DCMAKE_CUDA_ARCHITECTURES=<code> for another GPU.
if(NOT CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES is not set")
endif()

message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CUDA: ${CMAKE_CUDA_COMPILER_VERSION} arch=sm_${CMAKE_CUDA_ARCHITECTURES}")
