#include "serial_sender.h"
#include <iostream>
#include <windows.h>
#include <fcntl.h>
#include <io.h>

SerialSender::SerialSender(const std::string& port, int baudrate, int interval_ms)
    : port_(port), baudrate_(baudrate), interval_ms_(interval_ms),
      serial_handle_(INVALID_HANDLE_VALUE), is_running_(false) {
    // 设置控制台输出编码为 UTF-8
    SetConsoleOutputCP(CP_UTF8);
    _setmode(_fileno(stdout), _O_U8TEXT);
}

SerialSender::~SerialSender() {
    stop();
}

bool SerialSender::start() {
    if (!openSerialPort()) {
        return false;
    }
    
    std::cout << "串口已打开: " << port_ << std::endl;
    std::cout << "发送间隔: " << interval_ms_ << "ms" << std::endl;
    
    is_running_ = true;
    send_thread_ = std::thread(&SerialSender::sendTask, this);
    
    return true;
}

void SerialSender::stop() {
    is_running_ = false;
    
    if (send_thread_.joinable()) {
        send_thread_.join();
    }
    
    closeSerialPort();
}

void SerialSender::sendTask() {
    // 测试数据
    const std::string test_data = "ATasdfasadfasdfsgsdgsdfeXXX 3213412 fsff";
    DWORD bytes_written;
    
    while (is_running_) {
        // 发送数据
        if (!WriteFile(serial_handle_, test_data.c_str(), test_data.length(), &bytes_written, nullptr)) {
            std::cerr << "发送失败: " << GetLastError() << std::endl;
            break;
        }
        
        // 等待指定时间
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
    }
}

bool SerialSender::openSerialPort() {
    // 打开串口
    std::string port_name = "\\\\.\\" + port_;
    serial_handle_ = CreateFileA(port_name.c_str(),
                               GENERIC_WRITE,
                               0,
                               nullptr,
                               OPEN_EXISTING,
                               0,
                               nullptr);
                               
    if (serial_handle_ == INVALID_HANDLE_VALUE) {
        std::cerr << "无法打开串口: " << port_ << std::endl;
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
    
    if (!SetCommState(serial_handle_, &dcb)) {
        std::cerr << "设置串口配置失败" << std::endl;
        CloseHandle(serial_handle_);
        return false;
    }
    
    // 设置超时
    COMMTIMEOUTS timeouts = {0};
    timeouts.WriteTotalTimeoutConstant = 50;
    timeouts.WriteTotalTimeoutMultiplier = 10;
    
    if (!SetCommTimeouts(serial_handle_, &timeouts)) {
        std::cerr << "设置超时参数失败" << std::endl;
        CloseHandle(serial_handle_);
        return false;
    }
    
    return true;
}

void SerialSender::closeSerialPort() {
    if (serial_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(serial_handle_);
        serial_handle_ = INVALID_HANDLE_VALUE;
        std::cout << "串口已关闭" << std::endl;
    }
} 