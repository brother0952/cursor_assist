from machine import Pin
import neopixel
import time
import math
import random
import network
import socket
import json

# 配置参数
NUM_LEDS = 10          # LED数量
PIN_NUM = 4            # GPIO引脚
np = neopixel.NeoPixel(Pin(PIN_NUM), NUM_LEDS)

# Wi-Fi配置
WIFI_SSID = "你的WiFi名称"
WIFI_PASSWORD = "你的WiFi密码"

# 颜色定义
COLORS = [
    (255, 0, 0),      # 红
    (0, 255, 0),      # 绿
    (0, 0, 255),      # 蓝
    (255, 255, 0),    # 黄
    (255, 0, 255),    # 紫
    (0, 255, 255),    # 青
    (255, 255, 255),  # 白
]

# 全局变量
current_effect = None
effect_running = False
stop_effect = False

def connect_wifi():
    """连接Wi-Fi"""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    
    if not wlan.isconnected():
        print(f"正在连接到Wi-Fi: {WIFI_SSID}")
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)
        
        # 等待连接
        for _ in range(20):  # 最多等待10秒
            if wlan.isconnected():
                break
            time.sleep(0.5)
    
    if wlan.isconnected():
        print(f"Wi-Fi连接成功!")
        print(f"IP地址: {wlan.ifconfig()[0]}")
        return wlan.ifconfig()[0]
    else:
        print("Wi-Fi连接失败!")
        return None

def clear():
    """清空所有LED"""
    for i in range(NUM_LEDS):
        np[i] = (0, 0, 0)
    np.write()

def set_color(r, g, b):
    """设置所有LED为指定颜色"""
    for i in range(NUM_LEDS):
        np[i] = (r, g, b)
    np.write()

def set_led(index, r, g, b):
    """设置单个LED颜色"""
    if 0 <= index < NUM_LEDS:
        np[index] = (r, g, b)
        np.write()

def wheel(pos):
    """生成彩虹色轮颜色"""
    if pos < 85:
        return (pos * 3, 255 - pos * 3, 0)
    elif pos < 170:
        pos -= 85
        return (255 - pos * 3, 0, pos * 3)
    else:
        pos -= 170
        return (0, pos * 3, 255 - pos * 3)

def stop_current_effect():
    """停止当前运行的特效"""
    global effect_running, stop_effect
    stop_effect = True
    while effect_running:
        time.sleep(0.01)  # 等待特效停止
    clear()

# ========== 特效函数 ==========
def effect_rainbow_cycle(speed=30, cycles=1):
    """彩虹循环效果"""
    global effect_running, stop_effect
    
    stop_effect = False
    effect_running = True
    
    try:
        for _ in range(cycles):
            for j in range(256):
                if stop_effect:
                    break
                for i in range(NUM_LEDS):
                    rc_index = (i * 256 // NUM_LEDS) + j
                    color = wheel(rc_index & 255)
                    np[i] = color
                np.write()
                time.sleep_ms(speed)
            if stop_effect:
                break
    finally:
        effect_running = False

def effect_breathing(color=(255, 0, 0), speed=10, cycles=3):
    """呼吸灯效果"""
    global effect_running, stop_effect
    
    stop_effect = False
    effect_running = True
    
    try:
        for _ in range(cycles):
            if stop_effect:
                break
            # 渐亮
            for brightness in range(0, 101, 5):
                if stop_effect:
                    break
                r = int(color[0] * brightness / 100)
                g = int(color[1] * brightness / 100)
                b = int(color[2] * brightness / 100)
                for i in range(NUM_LEDS):
                    np[i] = (r, g, b)
                np.write()
                time.sleep_ms(speed)
            
            if stop_effect:
                break
                
            # 渐暗
            for brightness in range(100, -1, -5):
                if stop_effect:
                    break
                r = int(color[0] * brightness / 100)
                g = int(color[1] * brightness / 100)
                b = int(color[2] * brightness / 100)
                for i in range(NUM_LEDS):
                    np[i] = (r, g, b)
                np.write()
                time.sleep_ms(speed)
    finally:
        effect_running = False

def effect_running_lights(color=(0, 0, 255), speed=100):
    """跑马灯效果"""
    global effect_running, stop_effect
    
    stop_effect = False
    effect_running = True
    
    try:
        while not stop_effect:
            for pos in range(NUM_LEDS):
                if stop_effect:
                    break
                clear()
                np[pos] = color
                if pos > 0:
                    np[pos-1] = tuple(c // 2 for c in color)
                if pos > 1:
                    np[pos-2] = tuple(c // 4 for c in color)
                np.write()
                time.sleep_ms(speed)
    finally:
        effect_running = False

def effect_police_lights(speed=100, cycles=10):
    """警灯效果"""
    global effect_running, stop_effect
    
    stop_effect = False
    effect_running = True
    
    try:
        for _ in range(cycles):
            if stop_effect:
                break
            # 红蓝交替
            for i in range(NUM_LEDS):
                if i % 2 == 0:
                    np[i] = (255, 0, 0)
                else:
                    np[i] = (0, 0, 255)
            np.write()
            time.sleep_ms(speed)
            
            if stop_effect:
                break
                
            # 蓝红交替
            for i in range(NUM_LEDS):
                if i % 2 == 0:
                    np[i] = (0, 0, 255)
                else:
                    np[i] = (255, 0, 0)
            np.write()
            time.sleep_ms(speed)
    finally:
        effect_running = False

def effect_fire(speed=50, duration=10):
    """火焰效果"""
    global effect_running, stop_effect
    
    stop_effect = False
    effect_running = True
    
    try:
        end_time = time.time() + duration
        while time.time() < end_time and not stop_effect:
            for i in range(NUM_LEDS):
                intensity = random.getrandbits(8)
                r = intensity
                g = random.getrandbits(7)
                b = random.getrandbits(5)
                np[i] = (r, g, b)
            np.write()
            time.sleep_ms(speed)
    finally:
        effect_running = False

def effect_meteor(color=(100, 200, 255), speed=40, decay=0.7):
    """流星效果"""
    global effect_running, stop_effect
    
    stop_effect = False
    effect_running = True
    
    try:
        while not stop_effect:
            for start in range(NUM_LEDS * 2):
                if stop_effect:
                    break
                for i in range(NUM_LEDS):
                    brightness = (i + 1) / NUM_LEDS
                    if start - i > 0 and start - i < NUM_LEDS:
                        intensity = brightness * 255 * (decay ** (NUM_LEDS - i - 1))
                        np[i] = tuple(int(c * intensity / 255) for c in color)
                    else:
                        np[i] = (0, 0, 0)
                np.write()
                time.sleep_ms(speed)
    finally:
        effect_running = False

def effect_gradient(speed=20):
    """渐变效果"""
    global effect_running, stop_effect
    
    stop_effect = False
    effect_running = True
    
    try:
        while not stop_effect:
            for i in range(len(COLORS)-1):
                if stop_effect:
                    break
                color1 = COLORS[i]
                color2 = COLORS[i+1]
                
                for step in range(101):
                    if stop_effect:
                        break
                    ratio = step / 100
                    r = int(color1[0] * (1-ratio) + color2[0] * ratio)
                    g = int(color1[1] * (1-ratio) + color2[1] * ratio)
                    b = int(color1[2] * (1-ratio) + color2[2] * ratio)
                    
                    for led in range(NUM_LEDS):
                        np[led] = (r, g, b)
                    np.write()
                    time.sleep_ms(speed)
    finally:
        effect_running = False

def effect_pulse(color=(0, 255, 255), speed=30):
    """脉冲效果"""
    global effect_running, stop_effect
    
    stop_effect = False
    effect_running = True
    
    try:
        while not stop_effect:
            for pos in range(NUM_LEDS):
                if stop_effect:
                    break
                for i in range(NUM_LEDS):
                    distance = abs(i - pos)
                    if distance < 3:
                        intensity = max(0, 255 - distance * 80)
                        np[i] = tuple(int(c * intensity / 255) for c in color)
                    else:
                        np[i] = (0, 0, 0)
                np.write()
                time.sleep_ms(speed)
    finally:
        effect_running = False

def effect_twinkle(speed=100, duration=10):
    """闪烁效果"""
    global effect_running, stop_effect
    
    stop_effect = False
    effect_running = True
    
    try:
        end_time = time.time() + duration
        while time.time() < end_time and not stop_effect:
            for i in range(NUM_LEDS):
                if random.getrandbits(1):
                    color = COLORS[random.getrandbits(3) % len(COLORS)]
                    np[i] = color
                else:
                    np[i] = (0, 0, 0)
            np.write()
            time.sleep_ms(speed)
    finally:
        effect_running = False

# 特效映射表
EFFECTS = {
    "rainbow": effect_rainbow_cycle,
    "breathing": effect_breathing,
    "running": effect_running_lights,
    "police": effect_police_lights,
    "fire": effect_fire,
    "meteor": effect_meteor,
    "gradient": effect_gradient,
    "pulse": effect_pulse,
    "twinkle": effect_twinkle,
}

# ========== HTTP服务器 ==========
def parse_request(request):
    """解析HTTP请求"""
    lines = request.split('\r\n')
    if not lines:
        return None, None, None
    
    # 解析请求行
    first_line = lines[0].split()
    if len(first_line) < 2:
        return None, None, None
    
    method = first_line[0]
    path = first_line[1]
    
    # 解析查询参数
    if '?' in path:
        path, query = path.split('?', 1)
    else:
        query = ''
    
    # 解析请求体
    body = None
    empty_line = False
    for i, line in enumerate(lines):
        if line == '':
            if i + 1 < len(lines):
                body = '\r\n'.join(lines[i+1:])
            break
    
    return method, path, query, body

def handle_api_request(method, path, query, body):
    """处理API请求"""
    global current_effect
    
    # 解析路径
    if path.startswith('/api/'):
        endpoint = path[5:]  # 移除 '/api/'
    else:
        endpoint = path[1:] if path.startswith('/') else path
    
    # 处理请求体
    params = {}
    if body:
        try:
            params = json.loads(body)
        except:
            # 尝试解析表单数据
            try:
                for pair in body.split('&'):
                    if '=' in pair:
                        key, value = pair.split('=', 1)
                        params[key] = value
            except:
                pass
    
    # 解析查询参数
    if query:
        for pair in query.split('&'):
            if '=' in pair:
                key, value = pair.split('=', 1)
                params[key] = value
    
    print(f"API请求: {endpoint}, 参数: {params}")
    
    # 路由处理
    if endpoint == '' or endpoint == 'index':
        return handle_index()
    
    elif endpoint == 'status':
        return handle_status()
    
    elif endpoint == 'effects':
        return handle_effects_list()
    
    elif endpoint == 'effect/start':
        return handle_effect_start(method, params)
    
    elif endpoint == 'effect/stop':
        return handle_effect_stop()
    
    elif endpoint == 'color':
        return handle_color(method, params)
    
    elif endpoint == 'led':
        return handle_led(method, params)
    
    else:
        return create_response(404, {"error": "Not Found"})

def handle_index():
    """处理首页请求"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WS2812 LED控制器</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .section { margin: 20px 0; padding: 15px; background: #f8f8f8; border-radius: 5px; }
            .effect-btn { margin: 5px; padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; }
            .effect-btn:hover { background: #45a049; }
            .stop-btn { background: #f44336; }
            .stop-btn:hover { background: #d32f2f; }
            .color-picker { margin: 10px 0; }
            .status { padding: 10px; background: #e8f5e9; border-radius: 5px; margin: 10px 0; }
            .led-control { display: inline-block; margin: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>WS2812 LED控制器</h1>
            
            <div class="section">
                <h2>状态</h2>
                <div id="status" class="status">正在加载...</div>
                <button onclick="updateStatus()">刷新状态</button>
            </div>
            
            <div class="section">
                <h2>特效控制</h2>
                <div id="effects">
                    <button class="effect-btn" onclick="startEffect('rainbow')">彩虹循环</button>
                    <button class="effect-btn" onclick="startEffect('breathing')">呼吸灯</button>
                    <button class="effect-btn" onclick="startEffect('running')">跑马灯</button>
                    <button class="effect-btn" onclick="startEffect('police')">警灯效果</button>
                    <button class="effect-btn" onclick="startEffect('fire')">火焰效果</button>
                    <button class="effect-btn" onclick="startEffect('meteor')">流星雨</button>
                    <button class="effect-btn" onclick="startEffect('gradient')">渐变过渡</button>
                    <button class="effect-btn" onclick="startEffect('pulse')">脉冲波浪</button>
                    <button class="effect-btn" onclick="startEffect('twinkle')">闪烁效果</button>
                </div>
                <button class="effect-btn stop-btn" onclick="stopEffect()">停止特效</button>
            </div>
            
            <div class="section">
                <h2>颜色控制</h2>
                <div class="color-picker">
                    <input type="color" id="colorPicker" value="#ff0000">
                    <input type="range" id="brightness" min="0" max="100" value="100">
                    <span id="brightnessValue">100%</span>
                    <button onclick="setSolidColor()">设置颜色</button>
                    <button onclick="turnOff()">关闭所有灯</button>
                </div>
            </div>
            
            <div class="section">
                <h2>单个LED控制</h2>
                <div id="ledControls">
                    <!-- LED控制按钮会动态生成 -->
                </div>
            </div>
        </div>
        
        <script>
            function updateStatus() {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        let statusDiv = document.getElementById('status');
                        statusDiv.innerHTML = `
                            <strong>当前特效:</strong> ${data.current_effect || '无'}<br>
                            <strong>运行状态:</strong> ${data.effect_running ? '运行中' : '停止'}<br>
                            <strong>LED数量:</strong> ${data.num_leds}<br>
                            <strong>IP地址:</strong> ${data.ip_address || '未知'}
                        `;
                    });
            }
            
            function startEffect(effectName) {
                let speed = prompt('请输入速度 (1-100, 默认30):', '30');
                let color = prompt('请输入颜色 (RGB格式如 255,0,0 或留空使用默认):', '');
                
                let params = { effect: effectName };
                if (speed) params.speed = parseInt(speed);
                if (color) {
                    let rgb = color.split(',').map(Number);
                    if (rgb.length === 3) {
                        params.color = { r: rgb[0], g: rgb[1], b: rgb[2] };
                    }
                }
                
                fetch('/api/effect/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(params)
                })
                .then(response => response.json())
                .then(data => {
                    alert(data.message || '特效已启动');
                    updateStatus();
                });
            }
            
            function stopEffect() {
                fetch('/api/effect/stop', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        alert(data.message);
                        updateStatus();
                    });
            }
            
            function setSolidColor() {
                let colorHex = document.getElementById('colorPicker').value;
                let brightness = document.getElementById('brightness').value;
                
                // 将HEX转换为RGB
                let r = parseInt(colorHex.slice(1, 3), 16);
                let g = parseInt(colorHex.slice(3, 5), 16);
                let b = parseInt(colorHex.slice(5, 7), 16);
                
                // 应用亮度
                r = Math.floor(r * brightness / 100);
                g = Math.floor(g * brightness / 100);
                b = Math.floor(b * brightness / 100);
                
                fetch('/api/color', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ r: r, g: g, b: b })
                })
                .then(response => response.json())
                .then(data => {
                    alert('颜色已设置');
                });
            }
            
            function turnOff() {
                fetch('/api/color', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ r: 0, g: 0, b: 0 })
                })
                .then(response => response.json())
                .then(data => {
                    alert('所有灯已关闭');
                });
            }
            
            function generateLEDControls() {
                let container = document.getElementById('ledControls');
                container.innerHTML = '';
                
                for (let i = 0; i < 10; i++) {
                    let ledDiv = document.createElement('div');
                    ledDiv.className = 'led-control';
                    ledDiv.innerHTML = `
                        LED ${i}: 
                        <input type="color" id="led${i}Color" value="#ff0000">
                        <button onclick="setLED(${i})">设置</button>
                        <button onclick="turnOffLED(${i})">关闭</button>
                    `;
                    container.appendChild(ledDiv);
                }
            }
            
            function setLED(index) {
                let colorHex = document.getElementById('led' + index + 'Color').value;
                let r = parseInt(colorHex.slice(1, 3), 16);
                let g = parseInt(colorHex.slice(3, 5), 16);
                let b = parseInt(colorHex.slice(5, 7), 16);
                
                fetch('/api/led', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ index: index, r: r, g: g, b: b })
                });
            }
            
            function turnOffLED(index) {
                fetch('/api/led', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ index: index, r: 0, g: 0, b: 0 })
                });
            }
            
            // 初始化
            document.getElementById('brightness').addEventListener('input', function() {
                document.getElementById('brightnessValue').textContent = this.value + '%';
            });
            
            updateStatus();
            generateLEDControls();
        </script>
    </body>
    </html>
    """
    return create_response(200, html, content_type="text/html")

def handle_status():
    """获取状态"""
    from network import WLAN, STA_IF
    wlan = WLAN(STA_IF)
    ip_address = wlan.ifconfig()[0] if wlan.isconnected() else "未连接"
    
    status = {
        "current_effect": current_effect,
        "effect_running": effect_running,
        "num_leds": NUM_LEDS,
        "ip_address": ip_address,
        "colors_available": len(COLORS)
    }
    return create_response(200, status)

def handle_effects_list():
    """获取可用特效列表"""
    effects_list = {
        "effects": list(EFFECTS.keys()),
        "descriptions": {
            "rainbow": "彩虹循环效果",
            "breathing": "呼吸灯效果",
            "running": "跑马灯效果",
            "police": "警灯效果",
            "fire": "火焰效果",
            "meteor": "流星效果",
            "gradient": "渐变效果",
            "pulse": "脉冲波浪效果",
            "twinkle": "闪烁效果"
        }
    }
    return create_response(200, effects_list)

def handle_effect_start(method, params):
    """启动特效"""
    global current_effect
    
    if method != 'POST':
        return create_response(405, {"error": "Method not allowed"})
    
    if 'effect' not in params:
        return create_response(400, {"error": "Effect name required"})
    
    effect_name = params['effect']
    if effect_name not in EFFECTS:
        return create_response(400, {"error": "Effect not found"})
    
    # 停止当前特效
    stop_current_effect()
    
    # 准备参数
    effect_params = {}
    if 'speed' in params:
        effect_params['speed'] = int(params['speed'])
    if 'color' in params:
        color = params['color']
        if isinstance(color, dict) and 'r' in color and 'g' in color and 'b' in color:
            effect_params['color'] = (color['r'], color['g'], color['b'])
    if 'cycles' in params:
        effect_params['cycles'] = int(params['cycles'])
    if 'duration' in params:
        effect_params['duration'] = int(params['duration'])
    
    # 启动新特效（在新线程中运行）
    import _thread
    def run_effect():
        global current_effect
        current_effect = effect_name
        EFFECTS[effect_name](**effect_params)
        current_effect = None
    
    _thread.start_new_thread(run_effect, ())
    
    return create_response(200, {"message": f"Effect '{effect_name}' started", "params": effect_params})

def handle_effect_stop():
    """停止特效"""
    stop_current_effect()
    return create_response(200, {"message": "Effect stopped"})

def handle_color(method, params):
    """设置颜色"""
    if method != 'POST':
        return create_response(405, {"error": "Method not allowed"})
    
    if 'r' not in params or 'g' not in params or 'b' not in params:
        return create_response(400, {"error": "Color values required"})
    
    # 停止当前特效
    stop_current_effect()
    
    # 设置颜色
    r = int(params['r'])
    g = int(params['g'])
    b = int(params['b'])
    set_color(r, g, b)
    
    return create_response(200, {"message": "Color set", "color": {"r": r, "g": g, "b": b}})

def handle_led(method, params):
    """设置单个LED"""
    if method != 'POST':
        return create_response(405, {"error": "Method not allowed"})
    
    if 'index' not in params or 'r' not in params or 'g' not in params or 'b' not in params:
        return create_response(400, {"error": "LED index and color values required"})
    
    index = int(params['index'])
    r = int(params['r'])
    g = int(params['g'])
    b = int(params['b'])
    
    set_led(index, r, g, b)
    
    return create_response(200, {"message": f"LED {index} set", "color": {"r": r, "g": g, "b": b}})

def create_response(status_code, data, content_type="application/json"):
    """创建HTTP响应"""
    if content_type == "application/json":
        body = json.dumps(data)
    else:
        body = data
    
    response = f"HTTP/1.1 {status_code} {'OK' if status_code == 200 else 'Not Found'}\r\n"
    response += "Content-Type: " + content_type + "\r\n"
    response += "Access-Control-Allow-Origin: *\r\n"
    response += "Connection: close\r\n"
    response += "\r\n"
    response += body
    
    return response

def run_server():
    """运行HTTP服务器"""
    # 连接Wi-Fi
    ip_address = connect_wifi()
    if not ip_address:
        print("无法连接到Wi-Fi，请检查配置")
        return
    
    # 创建socket服务器
    addr = socket.getaddrinfo('0.0.0.0', 80)[0][-1]
    server_socket = socket.socket()
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(addr)
    server_socket.listen(5)
    
    print(f"服务器已启动，访问地址: http://{ip_address}")
    print("按下 Ctrl+C 停止服务器")
    
    try:
        while True:
            client_socket, client_addr = server_socket.accept()
            print(f"来自 {client_addr} 的连接")
            
            try:
                # 接收请求
                request = client_socket.recv(1024).decode('utf-8')
                if not request:
                    continue
                
                # 解析请求
                method, path, query, body = parse_request(request)
                
                # 处理请求
                if method and path:
                    response = handle_api_request(method, path, query, body)
                else:
                    response = create_response(400, {"error": "Invalid request"})
                
                # 发送响应
                client_socket.send(response.encode('utf-8'))
                
            except Exception as e:
                print(f"处理请求时出错: {e}")
                error_response = create_response(500, {"error": "Internal server error"})
                client_socket.send(error_response.encode('utf-8'))
                
            finally:
                client_socket.close()
                
    except KeyboardInterrupt:
        print("\n服务器停止")
    finally:
        server_socket.close()
        clear()

# 主程序
if __name__ == "__main__":
    # 清空LED
    clear()
    
    # 修改这里的Wi-Fi配置
    WIFI_SSID = "HUAWEI-P107NL"      # 修改为你的Wi-Fi名称
    WIFI_PASSWORD = "12841037"  # 修改为你的Wi-Fi密码
    
    # 运行服务器
    run_server()