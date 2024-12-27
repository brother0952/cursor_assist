#pragma once
#include <windows.h>
#include <string>
#include <fstream>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>

class SerialLogger {
public:
    SerialLogger(const std::string& port, int baudrate, double idle_threshold_ms = 5.0);
    ~SerialLogger();

    bool start();
    void stop();

private:
    // 串口参数
    std::string port_;
    int baudrate_;
    double idle_threshold_ms_;
    HANDLE serial_handle_;
    
    // 控制标志
    std::atomic<bool> is_running_;
    
    // 缓冲区和队列
    static const size_t BUFFER_SIZE = 8192;
    std::vector<uint8_t> current_frame_;
    std::queue<std::pair<std::string, std::vector<uint8_t>>> data_queue_;
    
    // 线程同步
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // 线程
    std::thread read_thread_;
    std::thread write_thread_;
    
    // 文件输出
    std::string log_file_;
    std::ofstream log_stream_;
    
    // 时间戳相关
    std::chrono::steady_clock::time_point last_receive_time_;
    
    // 线程函数
    void readTask();
    void writeTask();
    
    // 辅助函数
    std::string getCurrentTimestamp();
    bool openSerialPort();
    void closeSerialPort();

    LARGE_INTEGER freq_;  // 性能计数器频率
    LARGE_INTEGER start_time_;  // 启动时间点
    std::atomic<uint32_t> sequence_number_{0};

    OVERLAPPED read_overlapped_{0};  // 添加异步I/O结构
    HANDLE read_event_;              // 添加事件句柄
}; 