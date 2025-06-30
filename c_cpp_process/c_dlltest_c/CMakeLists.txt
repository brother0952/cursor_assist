cmake_minimum_required(VERSION 3.10)
project(MyDLL_C LANGUAGES C)

# 设置C标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 添加DLL库
add_library(MyDLL_C SHARED
    src/dllmain.c
)

# 包含目录
target_include_directories(MyDLL_C PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 添加测试子目录
add_subdirectory(test_app)
