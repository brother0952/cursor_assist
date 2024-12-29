# 使用默认参数
serial_logger.exe

# 指定串口和波特率
serial_logger.exe -p COM3 -b 115200

# 指定所有参数
serial_logger.exe -p COM7 -b 256000 -i 2.5 -o my_log.txt

# 显示帮助信息
serial_logger.exe --help

