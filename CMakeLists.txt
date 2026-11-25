cmake_minimum_required(VERSION 3.18)
project(cuda_vector_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optimised build by default: benchmarking a Debug build is meaningless.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# "native" targets the GPU installed in this machine (CMake >= 3.24).
# On older CMake, override: cmake -DCMAKE_CUDA_ARCHITECTURES=75 ..  (e.g. 75 = T4, 86 = RTX 30xx)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 75)
  endif()
endif()

add_executable(cuda_vector_benchmark
  src/main.cu
  src/gpu_vector_add.cu
  src/cpu_vector_add.cpp)

target_include_directories(cuda_vector_benchmark PRIVATE include)
