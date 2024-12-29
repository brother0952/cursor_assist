#include "serial_logger.h"
#include <iostream>
#include <csignal>
#include <string>
#include <cstdlib>

std::atomic<bool> g_running(true);

void signalHandler(int signum) {
    g_running = false;
}

void printUsage(const char* programName) {
    std::cout << "用法: " << programName << " [选项]" << std::endl;
    std::cout << "选项:" << std::endl;
    std::cout << "  -p, --port <串口号>       指定串口号 (默认: COM6)" << std::endl;
    std::cout << "  -b, --baudrate <波特率>   指定波特率 (默认: 500000)" << std::endl;
    std::cout << "  -i, --idle <毫秒>         指定空闲等待时间 (默认: 3.0ms)" << std::endl;
    std::cout << "  -o, --output <文件名>     指定输出文件名 (默认: 自动生成)" << std::endl;
    std::cout << "  -h, --help               显示此帮助信息" << std::endl;
    std::cout << std::endl;
    std::cout << "示例:" << std::endl;
    std::cout << "  " << programName << " -p COM3 -b 115200 -i 5.0" << std::endl;
    std::cout << "  " << programName << " --port COM7 --idle 2.0 --output my_log.txt" << std::endl;
}

int main(int argc, char* argv[]) {
    // 默认参数
    std::string port = "COM6";
    int baudrate = 500000;
    double idle_threshold = 3.0;
    std::string output_file;
    
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "-p" || arg == "--port") {
            if (i + 1 < argc) {
                port = argv[++i];
            }
        }
        else if (arg == "-b" || arg == "--baudrate") {
            if (i + 1 < argc) {
                baudrate = std::atoi(argv[++i]);
            }
        }
        else if (arg == "-i" || arg == "--idle") {
            if (i + 1 < argc) {
                idle_threshold = std::atof(argv[++i]);
            }
        }
        else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                output_file = argv[++i];
            }
        }
    }
    
    // 验证参数
    if (baudrate <= 0) {
        std::cerr << "错误: 无效的波特率" << std::endl;
        return 1;
    }
    
    if (idle_threshold <= 0) {
        std::cerr << "错误: 无效的空闲等待时间" << std::endl;
        return 1;
    }
    
    // 设置信号处理
    signal(SIGINT, signalHandler);
    
    // 创建串口记录器
    SerialLogger logger(port, baudrate, idle_threshold, output_file);
    
    if (!logger.start()) {
        return 1;
    }
    
    std::cout << "\n配置信息:" << std::endl;
    std::cout << "串口: " << port << std::endl;
    std::cout << "波特率: " << baudrate << std::endl;
    std::cout << "空闲等待时间: " << idle_threshold << "ms" << std::endl;
    std::cout << "\n正在记录数据... 按Ctrl+C停止" << std::endl;
    
    // 等待停止信号
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    std::cout << "\n停止记录..." << std::endl;
    logger.stop();
    std::cout << "记录完成" << std::endl;
    
    return 0;
} 