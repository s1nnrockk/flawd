cmake_minimum_required(VERSION 3.20)

project(FLAWD VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(flawd_engine engine/src/tensor.cpp)
target_include_directories(flawd_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/engine/include)

if(MSVC)
    target_compile_options(flawd_engine PRIVATE /W4 /permissive-)
else()
    target_compile_options(flawd_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(flawd_demo engine/src/main.cpp)
target_link_libraries(flawd_demo PRIVATE flawd_engine)
