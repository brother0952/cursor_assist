cmake_minimum_required(VERSION 3.10)
project(TestDLLApp_C LANGUAGES C)

# 设置C标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 添加可执行文件
add_executable(TestDLLApp_C
    src/main.c
)

# 包含DLL的头文件
target_include_directories(TestDLLApp_C PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# 链接DLL库
target_link_directories(TestDLLApp_C PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../build/lib
)

target_link_libraries(TestDLLApp_C PRIVATE
    MyDLL_C
)

# 复制DLL文件到输出目录
add_custom_command(TARGET TestDLLApp_C POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/../build/libMyDLL_C.dll
        $<TARGET_FILE_DIR:TestDLLApp_C>
    COMMENT "Copying DLL to output directory"
)

# 确保输出目录存在
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
