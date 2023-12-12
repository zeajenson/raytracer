cmake_minimum_required(VERSION 3.16.0)
project(raytracer CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "-O3 -mavx512f -mavx512dq -mavx512bw -mavx512vbmi -mavx512vbmi2 -mavx512vl")
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_BUILD_TYPE=Debug)

add_subdirectory("glfw")

add_executable(raytracer raytracer.cpp)

target_link_libraries(raytracer glfw)
