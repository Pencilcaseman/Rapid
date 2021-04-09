cmake_minimum_required (VERSION 3.8)
set(CMAKE_CUDA_STANDARD 17)

add_executable (CudaTest "cudaTest.cpp")

target_link_libraries(CudaTest PRIVATE rapid)
