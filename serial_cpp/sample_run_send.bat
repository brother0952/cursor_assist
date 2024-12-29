@echo off
REM 检查是否存在可执行文件
if not exist "build\serial_sender.exe" (
    echo 错误：找不到 serial_sender.exe
    echo 请先运行 compile.bat 编译程序
    pause
    exit /b 1
)

REM 运行程序
build\serial_sender.exe -p COM10 -b 500000 -i 7.0

pause 