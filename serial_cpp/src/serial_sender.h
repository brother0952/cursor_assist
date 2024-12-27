#pragma once
#include <windows.h>
#include <string>
#include <atomic>
#include <thread>

class SerialSender {
public:
    SerialSender(const std::string& port, int baudrate, int interval_ms = 7);
    ~SerialSender();

    bool start();
    void stop();

private:
    // 串口参数
    std::string port_;
    int baudrate_;
    int interval_ms_;
    HANDLE serial_handle_;
    
    // 控制标志
    std::atomic<bool> is_running_;
    
    // 发送线程
    std::thread send_thread_;
    
    // 线程函数
    void sendTask();
    
    // 辅助函数
    bool openSerialPort();
    void closeSerialPort();
}; 