cmake_minimum_required(VERSION 2.8)

project(main)

set(SOURCE main.cpp)
add_executable(main main.cpp)
target_sources(main PRIVATE ${HEADER})

add_subdirectory(neuro-network)
target_link_libraries(main PRIVATE neuro-network) #undefined symbols


