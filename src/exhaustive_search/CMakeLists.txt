cmake_minimum_required(VERSION 3.18)
project(cuda_sk_solver LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

include_directories(include)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(solver
	src/main.cpp
	src/solver.cu
)

set_target_properties(solver PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
)
