import network
import socket
import machine
import time

# 初始化 LED 引脚
led = machine.Pin(2, machine.Pin.OUT)

# 连接 Wi-Fi
ssid = "HUAWEI-P107NL"      # 修改为你的Wi-Fi名称
password = "12871034"  # 修改为你的Wi-Fi密码

wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(ssid, password)

# 等待连接
while not wlan.isconnected():
    time.sleep(1)

print('连接成功，IP 地址:', wlan.ifconfig()[0])

# 创建网页
html = """<!DOCTYPE html>
<html>
<head>
    <title>ESP8266 LED 控制</title>
</head>
<body>
    <h1>控制 LED</h1>
    <button onclick="fetch('/led/on')">打开 LED</button>
    <button onclick="fetch('/led/off')">关闭 LED</button>
    <h2>串口数据:</h2>
    <pre id="serial-data"></pre>
    <script>
        setInterval(() => {
            fetch('/serial').then(response => response.text()).then(data => {
                document.getElementById('serial-data').innerText = data;
            });
        }, 1000);
    </script>
</body>
</html>
"""

# 创建 socket
addr = socket.getaddrinfo('0.0.0.0', 80)[0]
s = socket.socket()
s.bind(addr)
s.listen(1)

print('监听中，等待连接...')

while True:
    cl, addr = s.accept()
    print('客户端连接来自:', addr)
    request = cl.recv(1024)
    request = str(request)

    if '/led/on' in request:
        led.on()
    elif '/led/off' in request:
        led.off()
    elif '/serial' in request:
        # 读取串口数据并返回
        serial_data = ''  # 在这里实现串口读取
        cl.send('HTTP/1.0 200 OK\r\nContent-type:text/html\r\n\r\n')
        cl.send(serial_data)
        cl.close()
        continue

    # 发送网页
    cl.send('HTTP/1.0 200 OK\r\nContent-type:text/html\r\n\r\n')
    cl.send(html)
    cl.close() 