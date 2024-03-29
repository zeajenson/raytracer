cmake_minimum_required(VERSION 3.16.0)
project(raytracer CXX)

set(CMAKE_CXX_STANDARD 23)
# set(CMAKE_CXX_FLAGS "-O3 -mavx512f -mavx512dq -mavx512bw -mavx512vbmi -mavx512vbmi2 -mavx512vl")
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_BUILD_TYPE=Debug)

add_subdirectory("glfw")

set(VERT_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/shader.vert")
set(FRAG_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/shader.frag")
set(COMP_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/shader.comp")

set(VERT_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shader.vert.spv")
set(FRAG_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shader.frag.spv")
set(COMP_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/shader.comp.spv")
add_custom_command( 
	OUTPUT ${VERT_OUTPUT} ${FRAG_OUTPUT} ${COMP_OUTPUT}
	# COMMAND glslc -fshader-stage=vert -o ${VERT_OUTPUT} ${VERT_SOURCE}\; glslc -fshader-stage=frag -o ${FRAG_OUTPUT} ${FRAG_SOURCE}
	COMMAND glslc -o ${VERT_OUTPUT} ${VERT_SOURCE}\; glslc -o ${FRAG_OUTPUT} ${FRAG_SOURCE}\; glslc -o ${COMP_OUTPUT} ${COMP_SOURCE}
	DEPENDS ${VERT_SOURCE} ${FRAG_SOURCE} ${COMP_SOURCE}
)
add_custom_target(shader_target DEPENDS ${VERT_OUTPUT} ${FRAG_OUTPUT})



add_executable(raytracer raytracer.cpp)

target_link_libraries(raytracer glfw)
add_dependencies(raytracer shader_target)
