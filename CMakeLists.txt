# it just for cmake-based ide or clangd to generate the compile_commands.json
cmake_minimum_required(VERSION 3.10)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
project(cuda_kernel_guide LANGUAGES CXX CUDA)

cmake_policy(SET CMP0148 NEW)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)

set(PYBIND11_FINDPYTHON ON)
set(PYBIND11_TEST OFF)
set(PYBIND11_INSTALL ON)

FetchContent_MakeAvailable(pybind11)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)

file(WRITE ${CMAKE_BINARY_DIR}/test_cuda.cu "extern \"C\" __global__ void testKernel() { }")
execute_process(
    COMMAND ${CUDA_NVCC_EXECUTABLE} ${CMAKE_CUDA_FLAGS} -gencode arch=compute_90a,code=sm_90a -o ${CMAKE_BINARY_DIR}/test_cuda.o -c ${CMAKE_BINARY_DIR}/test_cuda.cu
    RESULT_VARIABLE NVCC_RESULT
    OUTPUT_VARIABLE NVCC_OUTPUT
    ERROR_VARIABLE NVCC_ERROR_OUTPUT
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

if (NVCC_RESULT EQUAL "0")
    set(NVCC_SUPPORTS_SM90 TRUE)
    message(STATUS "NVCC supports SM90")
else()
    message(STATUS "NVCC does not support SM90")
endif()

if (NVCC_SUPPORTS_SM90)
    set(TORCH_CUDA_ARCH_LIST "8.6" CACHE STRING "Add arch tag 90a to NVCC" FORCE)
    list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_90a,code=sm_90a")
endif()

find_package(Torch REQUIRED)

include_directories(csrc/gemm 3rd/ThunderKittens/include 3rd/ThunderKittens/prototype 3rd/cutlass/include 3rd/cutlass/tools/util/include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
link_directories(${TORCH_INSTALL_PREFIX}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -fPIC")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -std=c++20 -DNDEBUG --ptxas-options=--register-usage-level=10")

cuda_add_library(example_gemm STATIC indexing/main.cu)
