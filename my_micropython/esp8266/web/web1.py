import network
import socket
import time

# Wi-Fi配置
SSID = "HUAWEI-P107NL"      # 修改为你的Wi-Fi名称
PASSWORD = "12871034"  # 修改为你的Wi-Fi密码

# 连接Wi-Fi
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(SSID, PASSWORD)

for i in range(20):
    if wlan.isconnected():
        break
    time.sleep(0.5)

if wlan.isconnected():
    ip = wlan.ifconfig()[0]
    print(f"IP地址: {ip}")
    
    # 创建服务器
    s = socket.socket()
    s.bind(('0.0.0.0', 80))
    s.listen(1)
    
    print("服务器已启动")
    
    while True:
        cl, addr = s.accept()
        print("客户端连接:", addr[0])
        
        request = cl.recv(1024)
        
        # 响应页面
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"> 
    <title>LED控制</title>
</head>
<body>
    <h1>ESP8266 LED控制器</h1>
    <p>中文测试：正常显示</p>
    <a href="/on">开灯</a>
    <a href="/off">关灯</a>
</body>
</html>"""
        
        # HTTP响应头
        response = "HTTP/1.1 200 OK\r\n"
        response += "Content-Type: text/html; charset=utf-8\r\n"  # 关键：这里设置编码
        response += "Connection: close\r\n"
        response += "\r\n"  # 空行分隔头部和正文
        response += html
        
        cl.send(response.encode('utf-8'))  # 使用UTF-8编码发送
        cl.close()