# 第三步：完整的最终版本（优化稳定）
import socket
import time
import network
from machine import Pin
import neopixel

# 配置 - 请修改！
WIFI_SSID = "HUAWEI-P107NL"      # 修改为你的Wi-Fi名称
WIFI_PASS = "12871034"  # 修改为你的Wi-Fi密码
LED_PIN = 4
LED_NUM = 10

# 初始化LED
leds = neopixel.NeoPixel(Pin(LED_PIN), LED_NUM)

def wifi_connect():
    """连接Wi-Fi"""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    
    if not wlan.isconnected():
        print(f"连接: {WIFI_SSID}")
        wlan.connect(WIFI_SSID, WIFI_PASS)
        
        wait = 0
        while not wlan.isconnected() and wait < 20:
            wait += 1
            time.sleep(0.5)
    
    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        print(f"IP: {ip}")
        return True
    return False

def led_off():
    for i in range(LED_NUM):
        leds[i] = (0, 0, 0)
    leds.write()

def led_color(r, g, b):
    for i in range(LED_NUM):
        leds[i] = (r, g, b)
    leds.write()

def web_server():
    """运行Web服务器"""
    if not wifi_connect():
        return
    
    s = socket.socket()
    s.bind(('0.0.0.0', 80))
    s.listen(1)
    
    print("Web服务器已启动")
    led_color(0, 0, 50)  # 蓝色表示就绪
    
    while True:
        try:
            cl, addr = s.accept()
            request = cl.recv(1024).decode()
            
            # 简单解析路径
            if 'GET / ' in request or 'GET /index' in request:
                response = """HTTP/1.1 200 OK\r\nContent-Type: text/html ;charset=utf-8\r\n\r\n
                <h1>LED控制</h1>
                <a href="/red">红</a>
                <a href="/green">绿</a>
                <a href="/blue">蓝</a>
                <a href="/white">白</a>
                <a href="/off">关</a>
                """
            
            elif 'GET /red' in request:
                led_color(255, 0, 0)
                response = "HTTP/1.1 200 OK\r\n\r\n红色"
            
            elif 'GET /green' in request:
                led_color(0, 255, 0)
                response = "HTTP/1.1 200 OK\r\n\r\n绿色"
            
            elif 'GET /blue' in request:
                led_color(0, 0, 255)
                response = "HTTP/1.1 200 OK\r\n\r\n蓝色"
            
            elif 'GET /white' in request:
                led_color(255, 255, 255)
                response = "HTTP/1.1 200 OK\r\n\r\n白色"
            
            elif 'GET /off' in request:
                led_off()
                response = "HTTP/1.1 200 OK\r\n\r\n关闭;charset=utf-8\r\n"
            
            else:
                response = "HTTP/1.1 404 Not Found\r\n\r\n404"
            
            cl.send(response.encode('utf-8'))
            cl.close()
            
        except Exception as e:
            print("错误:", e)
            try:
                cl.close()
            except:
                pass
            time.sleep(1)

# 运行
web_server()