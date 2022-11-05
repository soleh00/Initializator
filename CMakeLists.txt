cmake_minimum_required(VERSION 3.24)
#set(CUDACXX "/usr/local/cuda-11.8/bin/nvcc")
#set(CMAKE_CUDA_RUNTIME_LIBRARY "/usr/local/cuda-11.8/targets/x86_64-linux/lib")
project(NMR LANGUAGES CXX CUDA)

set(SOURCES src/main.cpp src/NmrRelaxationSolve.cpp src/NmrRelaxationSolveKernels.cu)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)


add_executable(NMR_exe ${SOURCES})
target_link_libraries(NMR_exe MPI::MPI_CXX)
target_link_libraries(NMR_exe CUDA::cudart)
set_target_properties(NMR_exe PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(NMR_exe PROPERTIES CUDA_ARCHITECTURES native)
