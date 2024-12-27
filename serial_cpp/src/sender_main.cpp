#include "serial_sender.h"
#include <iostream>
#include <csignal>

std::atomic<bool> g_running(true);

void signalHandler(int signum) {
    g_running = false;
}

int main() {
    // 设置信号处理
    signal(SIGINT, signalHandler);
    
    // 创建发送器
    SerialSender sender("COM10", 500000, 7);  // 7ms发送间隔
    
    if (!sender.start()) {
        return 1;
    }
    
    std::cout << "正在发送数据... 按Ctrl+C停止" << std::endl;
    
    // 等待停止信号
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    std::cout << "\n停止发送..." << std::endl;
    sender.stop();
    std::cout << "发送完成" << std::endl;
    
    return 0;
} 