cmake_minimum_required(VERSION 3.16.0)
project(raytracer CXX)

set(CMAKE_CXX_STANDARD 23)
# set(CMAKE_CXX_FLAGS "-O3 -mavx512f -mavx512dq -mavx512bw -mavx512vbmi -mavx512vbmi2 -mavx512vl")
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_BUILD_TYPE=Debug)

add_subdirectory("glfw")

set(VERT_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/shader.vert.glsl")
set(FRAG_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/shader.frag.glsl")
set(VERT_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shader.vert.spv")
set(FRAG_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shader.frag.spv")
add_custom_command( 
	OUTPUT ${VERT_OUTPUT} ${FRAG_OUTPUT}
	COMMAND glslc -fshader-stage=vert -o ${VERT_OUTPUT} ${VERT_SOURCE}\; glslc -fshader-stage=vert -o ${FRAG_OUTPUT} ${FRAG_SOURCE}
	DEPENDS ${VERT_SOURCE} ${FRAG_SOURCE}
)
add_custom_target(shader_target DEPENDS ${VERT_OUTPUT} ${FRAG_OUTPUT})



add_executable(raytracer raytracer.cpp)

target_link_libraries(raytracer glfw)
add_dependencies(raytracer shader_target)
