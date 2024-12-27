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
    
    // 初始化性能计数器
    QueryPerformanceFrequency(&freq_);
    QueryPerformanceCounter(&start_time_);
    
    // 创建事件句柄
    read_event_ = CreateEvent(nullptr, TRUE, FALSE, nullptr);
    read_overlapped_.hEvent = read_event_;
}

SerialLogger::~SerialLogger() {
    stop();
    if (read_event_) {
        CloseHandle(read_event_);
    }
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
    std::vector<uint8_t> buffer(64);
    std::vector<std::pair<std::string, std::vector<uint8_t>>> batch;
    DWORD bytes_read;
    
    last_receive_time_ = std::chrono::steady_clock::now();
    
    while (is_running_) {
        // 开始异步读取
        BOOL read_result = ReadFile(
            serial_handle_,
            buffer.data(),
            buffer.size(),
            nullptr,  // 异步操作不使用这个参数
            &read_overlapped_
        );
        
        DWORD error = GetLastError();
        if (!read_result && error != ERROR_IO_PENDING) {
            std::cerr << "读取失败: " << error << std::endl;
            break;
        }
        
        // 等待读取完成
        DWORD wait_result = WaitForSingleObject(read_event_, 100);  // 100ms超时
        if (wait_result == WAIT_OBJECT_0) {
            // 获取实际读取的字节数
            if (GetOverlappedResult(serial_handle_, &read_overlapped_, &bytes_read, FALSE)) {
                if (bytes_read > 0) {
                    for (DWORD i = 0; i < bytes_read; ++i) {
                        current_frame_.push_back(buffer[i]);
                        
                        if (buffer[i] == '\n') {
                            batch.emplace_back(getCurrentTimestamp(), current_frame_);
                            current_frame_.clear();
                            last_receive_time_ = std::chrono::steady_clock::now();
                        }
                    }
                    
                    if (!batch.empty()) {
                        std::lock_guard<std::mutex> lock(queue_mutex_);
                        for (auto& item : batch) {
                            data_queue_.push(std::move(item));
                        }
                        queue_cv_.notify_one();
                        batch.clear();
                    }
                }
            }
            
            // 重置事件，准备下一次读取
            ResetEvent(read_event_);
            
            // 重置overlapped结构的偏移量
            read_overlapped_.Offset = 0;
            read_overlapped_.OffsetHigh = 0;
        }
        else if (wait_result == WAIT_TIMEOUT) {
            // 超时，取消当前的I/O操作
            CancelIo(serial_handle_);
            continue;
        }
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
    static std::mutex timestamp_mutex;
    std::lock_guard<std::mutex> lock(timestamp_mutex);
    
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    
    // 计算从启动开始的时间（秒和纳秒）
    double elapsed = static_cast<double>(now.QuadPart - start_time_.QuadPart) / freq_.QuadPart;
    uint64_t seconds = static_cast<uint64_t>(elapsed);
    uint64_t nanos = static_cast<uint64_t>((elapsed - seconds) * 1000000000);
    
    // 获取系统时间作为基准
    auto sys_time = std::chrono::system_clock::now();
    auto sys_time_t = std::chrono::system_clock::to_time_t(sys_time);
    
    // 获取序列号
    uint32_t seq = sequence_number_++;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&sys_time_t), "%Y-%m-%d %H:%M:%S")
       << "." << std::setw(9) << std::setfill('0') << nanos
       << "_" << std::setw(4) << std::setfill('0') << (seq % 10000)
       << " (" << std::fixed << std::setprecision(3) << elapsed * 1000 << "ms)";  // 添加相对时间
    
    return ss.str();
}

bool SerialLogger::openSerialPort() {
    // 打开串口
    std::string port_name = "\\\\.\\" + port_;
    serial_handle_ = CreateFileA(
        port_name.c_str(),
        GENERIC_READ,
        0,
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_OVERLAPPED,  // 使用异步I/O
        nullptr
    );
    
    if (serial_handle_ == INVALID_HANDLE_VALUE) {
        std::cerr << "无法打开串口: " << port_ << std::endl;
        return false;
    }
    
    // 设置较小的缓冲区
    if (!SetupComm(serial_handle_, 64, 64)) {
        std::cerr << "设置串口缓冲区失败" << std::endl;
        CloseHandle(serial_handle_);
        return false;
    }
    
    // 配置串口参数
    DCB dcb = {0};
    dcb.DCBlength = sizeof(DCB);
    
    if (!GetCommState(serial_handle_, &dcb)) {
        std::cerr << "获取串口配置失败" << std::endl;
        CloseHandle(serial_handle_);
        return false;
    }
    
    dcb.BaudRate = baudrate_;
    dcb.ByteSize = 8;
    dcb.StopBits = ONESTOPBIT;
    dcb.Parity = NOPARITY;
    dcb.fBinary = TRUE;
    dcb.fDtrControl = DTR_CONTROL_DISABLE;
    dcb.fRtsControl = RTS_CONTROL_DISABLE;
    dcb.fOutX = FALSE;
    dcb.fInX = FALSE;
    dcb.fErrorChar = FALSE;
    dcb.fNull = FALSE;
    dcb.fAbortOnError = FALSE;
    
    if (!SetCommState(serial_handle_, &dcb)) {
        std::cerr << "设置串口配置失败" << std::endl;
        CloseHandle(serial_handle_);
        return false;
    }
    
    // 设置超时参数
    COMMTIMEOUTS timeouts = {0};
    timeouts.ReadIntervalTimeout = MAXDWORD;  // 立即返回
    timeouts.ReadTotalTimeoutMultiplier = 0;
    timeouts.ReadTotalTimeoutConstant = 0;
    
    if (!SetCommTimeouts(serial_handle_, &timeouts)) {
        std::cerr << "设置超时参数失败" << std::endl;
        CloseHandle(serial_handle_);
        return false;
    }
    
    // 清空缓冲区
    PurgeComm(serial_handle_, PURGE_RXCLEAR);
    
    return true;
}

void SerialLogger::closeSerialPort() {
    if (serial_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
    }
} 