cmake_minimum_required(VERSION 3.5)
project(ann)

set (CMAKE_C_COMPILER gcc)
set (CMAKE_CXX_COMPILER g++)

set(CMAKE_CXX_STANDARD 11)

include_directories(./include)

add_definitions("-g -w -O3")

aux_source_directory(./src SRC_LIST)

add_executable(ann ${SRC_LIST})

install(TARGETS ann RUNTIME DESTINATION ./)

