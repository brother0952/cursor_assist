@echo off
REM 检查是否存在可执行文件
if not exist "build\serial_logger.exe" (
    echo 错误：找不到 serial_logger.exe
    echo 请先运行 compile.bat 编译程序
    pause
    exit /b 1
)

REM 运行程序
@REM build\serial_logger.exe -p COM6 -b 500000 -i 4.0 -o my_log.txt
build\serial_logger.exe -p COM6 -b 256000 -i 4.0 -o my_log.txt
pause 