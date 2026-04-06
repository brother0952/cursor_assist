from machine import Pin
import neopixel
import time
import socket
import network

# 配置参数
NUM_LEDS = 10
PIN_NUM = 4
np = neopixel.NeoPixel(Pin(PIN_NUM), NUM_LEDS)

# Wi-Fi配置 - 请修改为你的Wi-Fi
WIFI_SSID = "your_wifi"
WIFI_PASSWORD = "your_password"

# 全局变量
effect_active = False

def connect_wifi():
    """连接Wi-Fi"""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    
    if not wlan.isconnected():
        print(f"连接Wi-Fi: {WIFI_SSID}")
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)
        
        for _ in range(20):  # 等待10秒
            if wlan.isconnected():
                break
            time.sleep(0.5)
    
    if wlan.isconnected():
        print(f"连接成功! IP: {wlan.ifconfig()[0]}")
        return wlan.ifconfig()[0]
    else:
        print("连接失败!")
        return None

def clear():
    """清空LED"""
    for i in range(NUM_LEDS):
        np[i] = (0, 0, 0)
    np.write()

def set_color(r, g, b):
    """设置所有LED颜色"""
    global effect_active
    effect_active = False
    for i in range(NUM_LEDS):
        np[i] = (r, g, b)
    np.write()

def set_single_led(index, r, g, b):
    """设置单个LED"""
    if 0 <= index < NUM_LEDS:
        np[index] = (r, g, b)
        np.write()

# ========== 轻量级特效 ==========
def effect_rainbow():
    """彩虹效果"""
    global effect_active
    effect_active = True
    
    while effect_active:
        for j in range(256):
            if not effect_active:
                break
            for i in range(NUM_LEDS):
                rc_index = (i * 256 // NUM_LEDS) + j
                if rc_index < 85:
                    np[i] = (rc_index * 3, 255 - rc_index * 3, 0)
                elif rc_index < 170:
                    rc_index -= 85
                    np[i] = (255 - rc_index * 3, 0, rc_index * 3)
                else:
                    rc_index -= 170
                    np[i] = (0, rc_index * 3, 255 - rc_index * 3)
            np.write()
            time.sleep_ms(30)
    clear()

def effect_breathing():
    """呼吸灯"""
    global effect_active
    effect_active = True
    
    while effect_active:
        # 红
        for brightness in range(0, 101, 5):
            if not effect_active: break
            color = (brightness * 255 // 100, 0, 0)
            for i in range(NUM_LEDS):
                np[i] = color
            np.write()
            time.sleep_ms(30)
        for brightness in range(100, -1, -5):
            if not effect_active: break
            color = (brightness * 255 // 100, 0, 0)
            for i in range(NUM_LEDS):
                np[i] = color
            np.write()
            time.sleep_ms(30)
        
        # 绿
        for brightness in range(0, 101, 5):
            if not effect_active: break
            color = (0, brightness * 255 // 100, 0)
            for i in range(NUM_LEDS):
                np[i] = color
            np.write()
            time.sleep_ms(30)
        for brightness in range(100, -1, -5):
            if not effect_active: break
            color = (0, brightness * 255 // 100, 0)
            for i in range(NUM_LEDS):
                np[i] = color
            np.write()
            time.sleep_ms(30)
        
        # 蓝
        for brightness in range(0, 101, 5):
            if not effect_active: break
            color = (0, 0, brightness * 255 // 100)
            for i in range(NED_LEDS):
                np[i] = color
            np.write()
            time.sleep_ms(30)
        for brightness in range(100, -1, -5):
            if not effect_active: break
            color = (0, 0, brightness * 255 // 100)
            for i in range(NUM_LEDS):
                np[i] = color
            np.write()
            time.sleep_ms(30)
    clear()

def effect_running():
    """跑马灯"""
    global effect_active
    effect_active = True
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = 0
    
    while effect_active:
        color = colors[color_index]
        color_index = (color_index + 1) % 3
        
        for pos in range(NUM_LEDS):
            if not effect_active:
                break
            clear()
            np[pos] = color
            np.write()
            time.sleep_ms(100)
    clear()

def effect_police():
    """警灯"""
    global effect_active
    effect_active = True
    
    while effect_active:
        # 红色闪烁
        for i in range(NUM_LEDS):
            if i % 2 == 0:
                np[i] = (255, 0, 0)
            else:
                np[i] = (0, 0, 0)
        np.write()
        if not effect_active: break
        time.sleep_ms(200)
        
        # 蓝色闪烁
        for i in range(NUM_LEDS):
            if i % 2 == 0:
                np[i] = (0, 0, 255)
            else:
                np[i] = (0, 0, 0)
        np.write()
        if not effect_active: break
        time.sleep_ms(200)
    clear()

def stop_effect():
    """停止特效"""
    global effect_active
    effect_active = False
    time.sleep(0.1)  # 等待特效线程停止
    clear()

# ========== 轻量级Web服务器 ==========
def handle_request(client_socket):
    """处理HTTP请求"""
    try:
        request = client_socket.recv(1024).decode('utf-8')
        if not request:
            return
        
        lines = request.split('\n')
        first_line = lines[0]
        
        if 'GET / ' in first_line or 'GET /index' in first_line:
            send_web_page(client_socket)
        
        elif 'GET /rainbow' in first_line:
            stop_effect()
            import _thread
            _thread.start_new_thread(effect_rainbow, ())
            send_response(client_socket, "彩虹效果已启动")
        
        elif 'GET /breathing' in first_line:
            stop_effect()
            import _thread
            _thread.start_new_thread(effect_breathing, ())
            send_response(client_socket, "呼吸灯已启动")
        
        elif 'GET /running' in first_line:
            stop_effect()
            import _thread
            _thread.start_new_thread(effect_running, ())
            send_response(client_socket, "跑马灯已启动")
        
        elif 'GET /police' in first_line:
            stop_effect()
            import _thread
            _thread.start_new_thread(effect_police, ())
            send_response(client_socket, "警灯效果已启动")
        
        elif 'GET /off' in first_line:
            stop_effect()
            send_response(client_socket, "已关闭")
        
        elif 'GET /color?' in first_line:
            # 解析颜色参数
            import urllib.parse
            query = first_line.split('?')[1].split(' ')[0]
            params = urllib.parse.parse_qs(query)
            
            if 'r' in params and 'g' in params and 'b' in params:
                stop_effect()
                r = int(params['r'][0])
                g = int(params['g'][0])
                b = int(params['b'][0])
                set_color(r, g, b)
                send_response(client_socket, f"颜色已设置为 RGB({r},{g},{b})")
            else:
                send_response(client_socket, "参数错误")
        
        else:
            send_response(client_socket, "未知命令")
            
    except Exception as e:
        print(f"处理请求错误: {e}")
        send_response(client_socket, "服务器错误")
    finally:
        client_socket.close()

def send_response(client_socket, message):
    """发送HTTP响应"""
    response = "HTTP/1.1 200 OK\r\n"
    response += "Content-Type: text/html; charset=utf-8\r\n"
    response += "Connection: close\r\n\r\n"
    response += f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>LED控制</title>
        <style>
            body {{ font-family: Arial, sans-serif; text-align: center; margin: 20px; }}
            h1 {{ color: #333; }}
            .btn {{ 
                display: inline-block; 
                margin: 10px; 
                padding: 15px 30px; 
                background: #4CAF50; 
                color: white; 
                text-decoration: none; 
                border-radius: 5px; 
                font-size: 18px; 
                min-width: 150px;
            }}
            .btn:hover {{ background: #45a049; }}
            .btn-stop {{ background: #f44336; }}
            .btn-stop:hover {{ background: #d32f2f; }}
            .color-btn {{ background: #2196F3; }}
            .color-btn:hover {{ background: #0b7dda; }}
            .status {{ 
                padding: 10px; 
                background: #e8f5e9; 
                border-radius: 5px; 
                margin: 20px auto; 
                max-width: 300px;
            }}
            .quick-colors {{ margin: 20px; }}
            .quick-color {{
                display: inline-block;
                width: 50px;
                height: 50px;
                margin: 5px;
                border-radius: 5px;
                cursor: pointer;
                border: 2px solid #ddd;
            }}
            .quick-color:hover {{ border-color: #333; }}
        </style>
        <script>
            function setColor(r, g, b) {{
                window.location.href = '/color?r=' + r + '&g=' + g + '&b=' + b;
            }}
            
            function updateColor() {{
                let r = document.getElementById('r').value;
                let g = document.getElementById('g').value;
                let b = document.getElementById('b').value;
                document.getElementById('colorPreview').style.backgroundColor = 
                    'rgb(' + r + ',' + g + ',' + b + ')';
                document.getElementById('rgbValue').innerText = 'RGB(' + r + ',' + g + ',' + b + ')';
            }}
        </script>
    </head>
    <body>
        <h1>WS2812 LED控制器</h1>
        
        <div class="status">
            <strong>状态:</strong> {message}<br>
            <strong>LED数量:</strong> {NUM_LEDS}
        </div>
        
        <h2>特效模式</h2>
        <a href="/rainbow" class="btn">彩虹效果</a><br>
        <a href="/breathing" class="btn">呼吸灯</a><br>
        <a href="/running" class="btn">跑马灯</a><br>
        <a href="/police" class="btn">警灯效果</a><br>
        
        <h2>颜色控制</h2>
        <div class="quick-colors">
            <div class="quick-color" style="background:#ff0000;" onclick="setColor(255,0,0)"></div>
            <div class="quick-color" style="background:#00ff00;" onclick="setColor(0,255,0)"></div>
            <div class="quick-color" style="background:#0000ff;" onclick="setColor(0,0,255)"></div>
            <div class="quick-color" style="background:#ffff00;" onclick="setColor(255,255,0)"></div>
            <div class="quick-color" style="background:#ff00ff;" onclick="setColor(255,0,255)"></div>
            <div class="quick-color" style="background:#00ffff;" onclick="setColor(0,255,255)"></div>
            <div class="quick-color" style="background:#ffffff;" onclick="setColor(255,255,255)"></div>
        </div>
        
        <div style="margin: 20px;">
            <div id="colorPreview" style="width:100px;height:100px;margin:0 auto 10px;background:#ff0000;border-radius:5px;"></div>
            <div id="rgbValue" style="margin-bottom:10px;">RGB(255,0,0)</div>
            
            <div style="display:inline-block; text-align:left;">
                <label>R: <input type="range" id="r" min="0" max="255" value="255" oninput="updateColor()"></label><br>
                <label>G: <input type="range" id="g" min="0" max="255" value="0" oninput="updateColor()"></label><br>
                <label>B: <input type="range" id="b" min="0" max="255" value="0" oninput="updateColor()"></label><br>
            </div><br>
            
            <button onclick="setColor(
                document.getElementById('r').value,
                document.getElementById('g').value,
                document.getElementById('b').value
            )" class="btn color-btn">设置颜色</button>
        </div>
        
        <h2>其他控制</h2>
        <a href="/off" class="btn btn-stop">关闭所有灯</a><br><br>
        
        <div style="margin-top: 30px; color: #666; font-size: 14px;">
            <p>ESP8266 WS2812控制器<br>10个LED灯带</p>
        </div>
    </body>
    </html>
    """
    client_socket.send(response.encode('utf-8'))

def send_web_page(client_socket):
    """发送Web控制页面"""
    send_response(client_socket, "就绪")

def run_server():
    """运行Web服务器"""
    # 连接Wi-Fi
    ip = connect_wifi()
    if not ip:
        print("Wi-Fi连接失败，无法启动服务器")
        return
    
    # 创建socket
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', 80))
    server.listen(5)
    
    print(f"服务器已启动: http://{ip}")
    print("等待连接...")
    
    # 清空LED
    clear()
    
    try:
        while True:
            client, addr = server.accept()
            print(f"来自 {addr} 的连接")
            handle_request(client)
            
    except KeyboardInterrupt:
        print("\n服务器停止")
    except Exception as e:
        print(f"服务器错误: {e}")
    finally:
        server.close()
        clear()

# 超轻量版本 - 如果还是内存不足
def ultra_light_server():
    """超轻量版本 - 最小化内存使用"""
    print("启动超轻量Web服务器...")
    
    # 连接Wi-Fi
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    time.sleep(5)
    
    if wlan.isconnected():
        ip = wlan.ifconfig()[0]
        print(f"IP: {ip}")
    else:
        print("Wi-Fi连接失败")
        return
    
    # 创建服务器
    server = socket.socket()
    server.bind(('0.0.0.0', 80))
    server.listen(1)
    
    clear()
    
    # 非常简单的路由
    routes = {
        '/': '首页',
        '/rainbow': '彩虹',
        '/red': '红色',
        '/green': '绿色',
        '/blue': '蓝色',
        '/off': '关闭',
    }
    
    while True:
        try:
            client, addr = server.accept()
            request = client.recv(1024).decode()
            
            # 解析请求
            first_line = request.split('\n')[0] if '\n' in request else request
            path = first_line.split(' ')[1] if ' ' in first_line else '/'
            
            print(f"请求: {path}")
            
            # 处理请求
            if path == '/':
                response = """HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n
                <h1>LED控制</h1>
                <a href="/rainbow">彩虹</a><br>
                <a href="/red">红色</a><br>
                <a href="/green">绿色</a><br>
                <a href="/blue">蓝色</a><br>
                <a href="/off">关闭</a>
                """
            
            elif path == '/rainbow':
                stop_effect()
                import _thread
                _thread.start_new_thread(effect_rainbow, ())
                response = "HTTP/1.1 200 OK\r\n\r\n彩虹已启动"
            
            elif path == '/red':
                stop_effect()
                set_color(255, 0, 0)
                response = "HTTP/1.1 200 OK\r\n\r\n红色已设置"
            
            elif path == '/green':
                stop_effect()
                set_color(0, 255, 0)
                response = "HTTP/1.1 200 OK\r\n\r\n绿色已设置"
            
            elif path == '/blue':
                stop_effect()
                set_color(0, 0, 255)
                response = "HTTP/1.1 200 OK\r\n\r\n蓝色已设置"
            
            elif path == '/off':
                stop_effect()
                response = "HTTP/1.1 200 OK\r\n\r\n已关闭"
            
            else:
                response = "HTTP/1.1 404 Not Found\r\n\r\n页面不存在"
            
            client.send(response.encode())
            client.close()
            
        except Exception as e:
            print(f"错误: {e}")
            try:
                client.close()
            except:
                pass

# 主程序
if __name__ == "__main__":
    # 修改为你的Wi-Fi
    WIFI_SSID = "你的WiFi名称"
    WIFI_PASSWORD = "你的WiFi密码"
    
    # 尝试运行完整服务器，如果内存不足则运行超轻量版本
    try:
        run_server()
    except MemoryError:
        print("内存不足，切换到超轻量版本...")
        ultra_light_server()