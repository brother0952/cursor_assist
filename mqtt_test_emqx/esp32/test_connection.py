import socket
import time

def test_connection():
    try:
        # 测试与EMQX服务器的连接
        addr_info = socket.getaddrinfo("broker.emqx.io", 1883)[0][-1]
        print(f"正在测试连接到: {addr_info}")
        
        s = socket.socket()
        s.settimeout(5.0)
        s.connect(addr_info)
        print("连接成功！")
        s.close()
        
    except Exception as e:
        print(f"连接测试失败: {e}")

# 测试网络连接
while True:
    test_connection()
    time.sleep(5) 