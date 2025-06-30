cmake_minimum_required(VERSION 3.10)
project(TestDLLApp LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加可执行文件
add_executable(TestDLLApp
    src/main.cpp
)

# 包含DLL的头文件
target_include_directories(TestDLLApp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# 链接DLL库
target_link_directories(TestDLLApp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../build
)

target_link_libraries(TestDLLApp PRIVATE
    MyDLL
)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 复制DLL文件到输出目录
add_custom_command(TARGET TestDLLApp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/libMyDLL.dll
        $<TARGET_FILE_DIR:TestDLLApp>
    COMMENT "Copying DLL to output directory"
)
