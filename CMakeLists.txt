cmake_minimum_required(VERSION 3.20)

project(FLAWD VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(flawd_engine
    engine/src/tensor.cpp
)
target_include_directories(flawd_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/engine/include)

if(MSVC)
    target_compile_options(flawd_engine PRIVATE /W4 /permissive-)
else()
    target_compile_options(flawd_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(flawd_demo engine/src/main.cpp)
target_link_libraries(flawd_demo PRIVATE flawd_engine)

include(CTest)
if(BUILD_TESTING)
    add_executable(flawd_engine_tests tests/cpp/test_tensor.cpp)
    target_link_libraries(flawd_engine_tests PRIVATE flawd_engine)
    add_test(NAME flawd_engine_tests COMMAND flawd_engine_tests)
endif()
