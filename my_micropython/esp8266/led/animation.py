from machine import Pin
import neopixel
import time
import math
import random

# 配置参数
NUM_LEDS = 10          # LED数量
PIN_NUM = 4            # GPIO引脚（根据你的连接修改）
np = neopixel.NeoPixel(Pin(PIN_NUM), NUM_LEDS)

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

def clear():
    """清空所有LED"""
    for i in range(NUM_LEDS):
        np[i] = (0, 0, 0)
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

def rainbow_cycle(wait=50):
    """彩虹循环效果"""
    for j in range(256):
        for i in range(NUM_LEDS):
            rc_index = (i * 256 // NUM_LEDS) + j
            color = wheel(rc_index & 255)
            np[i] = color
        np.write()
        time.sleep_ms(wait)

def breathing(color=(255, 0, 0), cycles=3, speed=10):
    """呼吸灯效果"""
    for _ in range(cycles):
        # 渐亮
        for brightness in range(0, 101, 5):
            r = int(color[0] * brightness / 100)
            g = int(color[1] * brightness / 100)
            b = int(color[2] * brightness / 100)
            for i in range(NUM_LEDS):
                np[i] = (r, g, b)
            np.write()
            time.sleep_ms(speed)
        
        # 渐暗
        for brightness in range(100, -1, -5):
            r = int(color[0] * brightness / 100)
            g = int(color[1] * brightness / 100)
            b = int(color[2] * brightness / 100)
            for i in range(NUM_LEDS):
                np[i] = (r, g, b)
            np.write()
            time.sleep_ms(speed)

def color_wave(wait=50):
    """彩色波浪效果"""
    for color in COLORS:
        for i in range(NUM_LEDS):
            np[i] = color
            np.write()
            time.sleep_ms(wait)
        time.sleep_ms(200)

def running_lights(color=(0, 0, 255), wait=100):
    """跑马灯效果"""
    for pos in range(NUM_LEDS):
        clear()
        np[pos] = color
        if pos > 0:
            np[pos-1] = tuple(c // 2 for c in color)  # 前一个灯半亮
        if pos > 1:
            np[pos-2] = tuple(c // 4 for c in color)  # 前两个灯更暗
        np.write()
        time.sleep_ms(wait)

def gradient_fade(wait=30):
    """渐变过渡效果"""
    for i in range(len(COLORS)-1):
        color1 = COLORS[i]
        color2 = COLORS[i+1]
        
        # 颜色渐变
        for step in range(101):
            ratio = step / 100
            r = int(color1[0] * (1-ratio) + color2[0] * ratio)
            g = int(color1[1] * (1-ratio) + color2[1] * ratio)
            b = int(color1[2] * (1-ratio) + color2[2] * ratio)
            
            for led in range(NUM_LEDS):
                np[led] = (r, g, b)
            np.write()
            time.sleep_ms(wait)

def fire_effect():
    """火焰效果 - MicroPython版本"""
    for _ in range(100):
        for i in range(NUM_LEDS):
            # 随机生成火焰颜色（红黄橙）
            intensity = random.getrandbits(8)  # 0-255随机数
            r = intensity
            g = random.getrandbits(7)  # 0-127
            b = random.getrandbits(5)  # 0-31
            np[i] = (r, g, b)
        np.write()
        time.sleep_ms(50 + random.getrandbits(6))  # 50-113ms随机延时

def police_lights(wait=100, cycles=10):
    """警灯效果"""
    for _ in range(cycles):
        # 红蓝交替
        for i in range(NUM_LEDS):
            if i % 2 == 0:
                np[i] = (255, 0, 0)  # 红灯
            else:
                np[i] = (0, 0, 255)  # 蓝灯
        np.write()
        time.sleep_ms(wait)
        
        # 蓝红交替
        for i in range(NUM_LEDS):
            if i % 2 == 0:
                np[i] = (0, 0, 255)  # 蓝灯
            else:
                np[i] = (255, 0, 0)  # 红灯
        np.write()
        time.sleep_ms(wait)

def meteor_rain(color=(255, 255, 255), wait=30, decay=0.7):
    """流星雨效果"""
    clear()
    
    for start in range(NUM_LEDS * 2):
        for i in range(NUM_LEDS):
            # 计算亮度衰减
            brightness = (i + 1) / NUM_LEDS
            
            if start - i > 0 and start - i < NUM_LEDS:
                intensity = brightness * 255 * (decay ** (NUM_LEDS - i - 1))
                np[i] = tuple(int(c * intensity / 255) for c in color)
            else:
                np[i] = (0, 0, 0)
        
        np.write()
        time.sleep_ms(wait)

def strobe_light(color=(255, 255, 255), wait=100, cycles=20):
    """闪光灯效果"""
    for _ in range(cycles):
        for i in range(NUM_LEDS):
            np[i] = color
        np.write()
        time.sleep_ms(wait)
        
        clear()
        time.sleep_ms(wait)

def pattern_effects():
    """图案效果 - 多种模式切换"""
    patterns = [
        [(255, 0, 0), (0, 0, 0)],  # 红黑交替
        [(0, 255, 0), (0, 0, 0)],  # 绿黑交替
        [(0, 0, 255), (0, 0, 0)],  # 蓝黑交替
        [(255, 255, 255), (0, 0, 0)],  # 白黑交替
        [(255, 0, 0), (0, 255, 0), (0, 0, 255)],  # RGB三色
    ]
    
    for pattern in patterns:
        for _ in range(5):  # 每个图案显示5次
            for i in range(NUM_LEDS):
                color = pattern[i % len(pattern)]
                np[i] = color
            np.write()
            time.sleep_ms(500)

def random_sparkle(wait=50, duration=5):
    """随机闪烁效果"""
    end_time = time.ticks_ms() + duration * 1000
    
    while time.ticks_ms() < end_time:
        # 随机点亮几个LED
        clear()
        num_sparkles = random.getrandbits(2) + 1  # 1-4个随机
        
        for _ in range(num_sparkles):
            led_pos = random.getrandbits(4) % NUM_LEDS  # 0-9随机位置
            color = COLORS[random.getrandbits(3) % len(COLORS)]  # 随机颜色
            np[led_pos] = color
        
        np.write()
        time.sleep_ms(wait)

def color_bounce(wait=100):
    """颜色弹跳效果"""
    colors = COLORS[:3]  # 使用前三种颜色
    
    for i in range(NUM_LEDS):
        for color in colors:
            clear()
            np[i] = color
            np[NUM_LEDS-1-i] = color
            np.write()
            time.sleep_ms(wait)

def dual_running_lights():
    """双色跑马灯效果"""
    colors = [(255, 0, 0), (0, 0, 255)]  # 红蓝双色
    
    for offset in range(NUM_LEDS):
        for i in range(NUM_LEDS):
            color_idx = (i + offset) % 2
            np[i] = colors[color_idx]
        np.write()
        time.sleep_ms(150)

def pulse_wave(color=(0, 255, 255), wait=30):
    """脉冲波浪效果"""
    for pos in range(NUM_LEDS):
        # 创建波浪
        for i in range(NUM_LEDS):
            distance = abs(i - pos)
            if distance < 3:
                intensity = max(0, 255 - distance * 80)
                np[i] = tuple(int(c * intensity / 255) for c in color)
            else:
                np[i] = (0, 0, 0)
        np.write()
        time.sleep_ms(wait)

def color_scanner(color=(255, 255, 0), wait=80):
    """扫描灯效果"""
    # 从左到右
    for pos in range(NUM_LEDS):
        clear()
        np[pos] = color
        np.write()
        time.sleep_ms(wait)
    
    # 从右到左
    for pos in range(NUM_LEDS-2, 0, -1):
        clear()
        np[pos] = color
        np.write()
        time.sleep_ms(wait)

def twinkle_colors(wait=100, duration=8):
    """五彩闪烁效果"""
    end_time = time.ticks_ms() + duration * 1000
    
    while time.ticks_ms() < end_time:
        for i in range(NUM_LEDS):
            if random.getrandbits(1):  # 50%概率点亮
                color = COLORS[random.getrandbits(3) % len(COLORS)]
                np[i] = color
            else:
                np[i] = (0, 0, 0)
        np.write()
        time.sleep_ms(wait)

def sine_wave(color=(255, 0, 255), speed=0.2):
    """正弦波效果"""
    import math
    
    offset = 0
    for _ in range(200):
        for i in range(NUM_LEDS):
            # 计算正弦波位置
            position = (i / NUM_LEDS) * 2 * math.pi + offset
            brightness = int((math.sin(position) + 1) * 127.5)  # 0-255
            
            np[i] = tuple(int(c * brightness / 255) for c in color)
        
        np.write()
        offset += speed
        time.sleep_ms(30)

# 简化版本的主函数，避免内存问题
def run_effects():
    """运行特效 - 简化的主循环"""
    effects = [
        ("彩虹循环", lambda: rainbow_cycle(wait=30)),
        ("呼吸灯", lambda: breathing(color=(255, 100, 0), cycles=2, speed=15)),
        ("彩色波浪", lambda: color_wave(wait=60)),
        ("渐变过渡", lambda: gradient_fade(wait=25)),
        ("跑马灯", lambda: running_lights(color=(0, 255, 255), wait=80)),
        ("火焰效果", lambda: fire_effect()),
        ("警灯", lambda: police_lights(cycles=8)),
        ("流星雨", lambda: meteor_rain(color=(100, 200, 255), wait=35)),
        ("扫描灯", lambda: color_scanner(color=(255, 255, 0), wait=70)),
        ("双色跑马灯", lambda: dual_running_lights()),
        ("脉冲波浪", lambda: pulse_wave(color=(255, 150, 0), wait=25)),
        ("五彩闪烁", lambda: twinkle_colors(wait=120, duration=5)),
    ]
    
    while True:
        try:
            for name, effect in effects:
                print(f"效果: {name}")
                effect()
                time.sleep_ms(500)  # 效果间暂停
                clear()
                
        except KeyboardInterrupt:
            print("\n程序停止")
            clear()
            break

# 直接运行效果，避免内存问题
print("WS2812 LED特效演示开始！")
run_effects()