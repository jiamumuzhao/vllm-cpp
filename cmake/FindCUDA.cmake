# FindCUDA.cmake
# CUDA toolkit detection and validation for vllm-cpp

if(NOT DEFINED CMAKE_CUDA_COMPILER)
  find_program(CMAKE_CUDA_COMPILER
    NAMES nvcc
    PATHS /usr/local/cuda/bin /opt/cuda/bin
  )
endif()

if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "nvcc not found. Install CUDA toolkit or set CMAKE_CUDA_COMPILER.")
endif()

execute_process(
  COMMAND ${CMAKE_CUDA_COMPILER} --version
  OUTPUT_VARIABLE NVCC_VERSION_OUT
  ERROR_VARIABLE  NVCC_VERSION_ERR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

string(REGEX MATCH "release [0-9]+\\.[0-9]+" NVCC_VER "${NVCC_VERSION_OUT}")
message(STATUS "Found CUDA compiler: ${CMAKE_CUDA_COMPILER} (${NVCC_VER})")

# Verify CUDA 13.x
string(REGEX MATCH "release ([0-9]+)\\.[0-9]+" _dummy "${NVCC_VERSION_OUT}")
if(NOT CMAKE_MATCH_1 STREQUAL "13")
  message(WARNING "Expected CUDA 13.x, found ${NVCC_VER}. Compatibility not guaranteed.")
endif()

# Locate CUDA toolkit root
get_filename_component(CUDA_TOOLKIT_ROOT "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(CUDA_TOOLKIT_ROOT "${CUDA_TOOLKIT_ROOT}" DIRECTORY)
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT} CACHE PATH "CUDA Toolkit root")

find_library(CUDA_CUDART_LIBRARY cudart PATHS "${CUDA_TOOLKIT_ROOT}/lib64" NO_DEFAULT_PATH)
find_library(CUDA_CUBLAS_LIBRARY cublas PATHS "${CUDA_TOOLKIT_ROOT}/lib64" NO_DEFAULT_PATH)
find_library(CUDA_CUBLASLT_LIBRARY cublasLt PATHS "${CUDA_TOOLKIT_ROOT}/lib64" NO_DEFAULT_PATH)

if(NOT CUDA_CUDART_LIBRARY)
  message(FATAL_ERROR "cudart not found in ${CUDA_TOOLKIT_ROOT}/lib64")
endif()

message(STATUS "CUDA toolkit: ${CUDA_TOOLKIT_ROOT}")
message(STATUS "  cudart:   ${CUDA_CUDART_LIBRARY}")
message(STATUS "  cublas:   ${CUDA_CUBLAS_LIBRARY}")
message(STATUS "  cublasLt: ${CUDA_CUBLASLT_LIBRARY}")
