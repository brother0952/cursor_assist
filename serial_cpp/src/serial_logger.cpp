#include "serial_logger.h"
#include <windows.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <direct.h>
#include <fcntl.h>
#include <io.h>

SerialLogger::SerialLogger(const std::string& port, int baudrate, double idle_threshold_ms)
    : port_(port), baudrate_(baudrate), idle_threshold_ms_(idle_threshold_ms),
      serial_handle_(INVALID_HANDLE_VALUE), is_running_(false) {
    // 设置控制台输出编码为 UTF-8
    SetConsoleOutputCP(CP_UTF8);
    _setmode(_fileno(stdout), _O_U8TEXT);
    
    // 创建日志目录
    _mkdir("logs");
    
    // 生成日志文件名
    auto now = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "logs/serial_log_" << std::put_time(std::localtime(&now_t), "%Y%m%d_%H%M%S") << ".txt";
    log_file_ = ss.str();
    
    // 预分配缓冲区
    current_frame_.reserve(BUFFER_SIZE);
}

SerialLogger::~SerialLogger() {
    stop();
}

bool SerialLogger::start() {
    if (!openSerialPort()) {
        return false;
    }
    
    // 打开日志文件
    log_stream_.open(log_file_);
    if (!log_stream_) {
        std::cerr << "无法创建日志文件: " << log_file_ << std::endl;
        closeSerialPort();
        return false;
    }
    
    std::cout << "串口已打开: " << port_ << std::endl;
    std::cout << "数据将保存到: " << log_file_ << std::endl;
    
    // 启动线程
    is_running_ = true;
    read_thread_ = std::thread(&SerialLogger::readTask, this);
    write_thread_ = std::thread(&SerialLogger::writeTask, this);
    
    return true;
}

void SerialLogger::stop() {
    is_running_ = false;
    
    if (read_thread_.joinable()) {
        read_thread_.join();
    }
    
    queue_cv_.notify_one();
    if (write_thread_.joinable()) {
        write_thread_.join();
    }
    
    closeSerialPort();
    
    if (log_stream_.is_open()) {
        log_stream_.close();
    }
}

void SerialLogger::readTask() {
    std::vector<uint8_t> buffer(BUFFER_SIZE);
    DWORD bytes_read;
    
    last_receive_time_ = std::chrono::steady_clock::now();
    
    while (is_running_) {
        BOOL success = ReadFile(serial_handle_, buffer.data(), buffer.size(), &bytes_read, nullptr);
        
        if (success && bytes_read > 0) {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - last_receive_time_).count();
            
            // 检查空闲时间
            if (!current_frame_.empty() && elapsed >= idle_threshold_ms_) {
                std::string timestamp = getCurrentTimestamp();
                
                std::lock_guard<std::mutex> lock(queue_mutex_);
                data_queue_.push(std::make_pair(timestamp, current_frame_));
                queue_cv_.notify_one();
                
                current_frame_.clear();
            }
            
            // 添加新数据
            current_frame_.insert(current_frame_.end(), buffer.data(), buffer.data() + bytes_read);
            last_receive_time_ = current_time;
        }
        
        // 短暂休眠
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void SerialLogger::writeTask() {
    while (is_running_ || !data_queue_.empty()) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        if (data_queue_.empty()) {
            queue_cv_.wait_for(lock, std::chrono::seconds(1));
            continue;
        }
        
        auto frame_data = std::move(data_queue_.front());
        data_queue_.pop();
        lock.unlock();
        
        // 写入数据
        log_stream_ << "[" << frame_data.first << "] ";
        for (const auto& byte : frame_data.second) {
            log_stream_ << std::uppercase << std::hex << std::setw(2) << std::setfill('0')
                       << static_cast<int>(byte) << " ";
        }
        log_stream_ << std::endl;
        log_stream_.flush();
    }
}

std::string SerialLogger::getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_t), "%Y-%m-%d %H:%M:%S")
       << "." << std::setw(3) << std::setfill('0') << now_ms.count();
    
    return ss.str();
}

bool SerialLogger::openSerialPort() {
    std::string port_name = "\\\\.\\" + port_;
    
    serial_handle_ = CreateFileA(
        port_name.c_str(),
        GENERIC_READ,
        0,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );
    
    if (serial_handle_ == INVALID_HANDLE_VALUE) {
        std::cerr << "无法打开串口: " << port_ << std::endl;
        return false;
    }
    
    DCB dcb = {0};
    dcb.DCBlength = sizeof(DCB);
    
    if (!GetCommState(serial_handle_, &dcb)) {
        std::cerr << "无法获取串口配置" << std::endl;
        CloseHandle(serial_handle_);
        return false;
    }
    
    dcb.BaudRate = baudrate_;
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    
    if (!SetCommState(serial_handle_, &dcb)) {
        std::cerr << "无法设置串口参数" << std::endl;
        CloseHandle(serial_handle_);
        return false;
    }
    
    COMMTIMEOUTS timeouts = {0};
    timeouts.ReadIntervalTimeout = 1;
    timeouts.ReadTotalTimeoutConstant = 0;
    timeouts.ReadTotalTimeoutMultiplier = 0;
    
    if (!SetCommTimeouts(serial_handle_, &timeouts)) {
        std::cerr << "无法设置串口超时" << std::endl;
        CloseHandle(serial_handle_);
        return false;
    }
    
    return true;
}

void SerialLogger::closeSerialPort() {
    if (serial_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
    }
} 