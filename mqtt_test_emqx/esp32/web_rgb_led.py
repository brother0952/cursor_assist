from machine import Pin
import neopixel
import network
import socket
import json

# WiFi配置
WIFI_SSID = "HUAWEI-P107NL"
WIFI_PASSWORD = "12871034"

# LED配置
LED_PIN = 38
LED_COUNT = 1
np = neopixel.NeoPixel(Pin(LED_PIN), LED_COUNT)

def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)
        while not wlan.isconnected():
            pass
    return wlan.ifconfig()[0]

def web_page():
    return """
    <html><body>
    <h1>ESP32 RGB LED Control</h1>
    <form action="/led" method="post">
        R: <input type="range" name="r" min="0" max="255"><br>
        G: <input type="range" name="g" min="0" max="255"><br>
        B: <input type="range" name="b" min="0" max="255"><br>
        <input type="submit" value="Set Color">
    </form>
    </body></html>
    """

def start_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 80))
    s.listen(5)

    while True:
        conn, addr = s.accept()
        request = conn.recv(1024)
        
        if b'POST /led' in request:
            # 处理LED控制请求
            try:
                # 解析POST数据
                data = request.split(b'\r\n\r\n')[1]
                params = dict(item.split('=') for item in data.decode().split('&'))
                r = int(params.get('r', 0))
                g = int(params.get('g', 0))
                b = int(params.get('b', 0))
                np[0] = (r, g, b)
                np.write()
            except:
                pass

        conn.send('HTTP/1.1 200 OK\n')
        conn.send('Content-Type: text/html\n')
        conn.send('Connection: close\n\n')
        conn.send(web_page())
        conn.close()

def main():
    ip = connect_wifi()
    print(f'Web server started on http://{ip}')
    start_server()

if __name__ == '__main__':
    main()
