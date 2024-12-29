@echo off
chcp 65001 > nul
setlocal

:: 设置 MinGW 路径
set MINGW_PATH=D:\Qt\Qt5.12.8\Tools\mingw730_32\bin
set PATH=%MINGW_PATH%;%PATH%

:: 创建构建目录
if not exist build mkdir build
cd build

:: 编译接收程序
echo 编译接收程序...
g++ -std=c++17 ^
    ..\src\main.cpp ^
    ..\src\serial_logger.cpp ^
    -o serial_logger.exe ^
    -I..\src ^
    -static-libgcc ^
    -static-libstdc++ ^
    -D_WIN32_WINNT=0x0601

:: 编译发送程序
echo 编译发送程序...
g++ -std=c++17 ^
    ..\src\sender_main.cpp ^
    ..\src\serial_sender.cpp ^
    -o serial_sender.exe ^
    -I..\src ^
    -static-libgcc ^
    -static-libstdc++ ^
    -D_WIN32_WINNT=0x0601

:: 检查编译结果
if %ERRORLEVEL% EQU 0 (
    echo 编译成功！
    echo 可执行文件位置:
    echo   接收程序: %CD%\serial_logger.exe
    echo   发送程序: %CD%\serial_sender.exe
) else (
    echo 编译失败！错误代码：%ERRORLEVEL%
)

:: 返回原目录
cd ..

endlocal 