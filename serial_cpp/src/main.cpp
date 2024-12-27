#include "serial_logger.h"
#include <iostream>
#include <csignal>

std::atomic<bool> g_running(true);

void signalHandler(int signum) {
    g_running = false;
}

int main() {
    // 设置信号处理
    signal(SIGINT, signalHandler);
    
    // 创建串口记录器
    SerialLogger logger("COM6", 500000, 3.0);  // 5ms空闲阈值
    
    if (!logger.start()) {
        return 1;
    }
    
    std::cout << "正在记录数据... 按Ctrl+C停止" << std::endl;
    
    // 等待停止信号
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    std::cout << "\n停止记录..." << std::endl;
    logger.stop();
    std::cout << "记录完成" << std::endl;
    
    return 0;
} 