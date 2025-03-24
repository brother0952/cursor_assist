from machine import Pin, PWM
import network
import time
from umqtt.simple import MQTTClient
import json
import random

import neopixel


# WiFi配置
WIFI_SSID = "HUAWEI-P107NL"
WIFI_PASSWORD = "12871034"

# MQTT配置
BROKER = "broker.emqx.io"
PORT = 1883
DEVICE_ID = f"smart_light_{random.randint(0, 1000)}"
DISCOVERY_PREFIX = "homeassistant"

# 主题
TOPIC_STATE = f"home/lights/{DEVICE_ID}/state"
TOPIC_SET = f"home/lights/{DEVICE_ID}/set"
TOPIC_AVAILABILITY = f"home/lights/{DEVICE_ID}/availability"
TOPIC_DISCOVERY = f"{DISCOVERY_PREFIX}/light/{DEVICE_ID}/config"

# LED配置（使用PWM模拟调光）
# led_pin = PWM(Pin(2))  # 根据实际连接调整
# led_pin.freq(1000)

# WS2812 LED配置
LED_PIN = 38  # WS2812数据引脚
LED_COUNT = 1  # LED数量
np = neopixel.NeoPixel(Pin(LED_PIN), LED_COUNT)

def set_led_color(temp):
    """根据温度设置LED颜色
    温度 < 25: 蓝色
    25 <= 温度 <= 30: 绿色
    温度 > 30: 红色
    """
    if temp > 30:
        np[0] = (temp, 0, 0)  # 红色 (R,G,B)
    elif temp >= 25:
        np[0] = (0, temp, 0)  # 绿色
    else:
        np[0] = (0, 0, temp)  # 蓝色
    np.write()  # 更新LED显示

def connect_wifi():
    """连接WiFi"""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print('连接到WiFi...')
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)
        while not wlan.isconnected():
            time.sleep(1)
    print('WiFi已连接')
    print('IP地址:', wlan.ifconfig()[0])

def set_brightness(brightness):
    """设置LED亮度 (0-100)"""
    # duty = int(brightness * 1023 / 100)
    # led_pin.duty(duty)
    set_led_color(brightness)

def on_message(topic, msg):
    """处理接收到的消息"""
    print(f'收到消息: {topic} {msg}')
    try:
        command = json.loads(msg.decode())
        if 'brightness' in command:
            brightness = command['brightness']
            set_brightness(brightness)
            publish_state(client, brightness)
    except:
        print('无效的命令格式')

def publish_state(client, brightness):
    """发布当前状态"""
    state = {
        "state": "ON" if brightness > 0 else "OFF",
        "brightness": brightness
    }
    client.publish(TOPIC_STATE, json.dumps(state))

def publish_discovery(client):
    """发布Home Assistant发现配置"""
    config = {
        "name": "Smart Light",
        "unique_id": DEVICE_ID,
        "command_topic": TOPIC_SET,
        "state_topic": TOPIC_STATE,
        "availability_topic": TOPIC_AVAILABILITY,
        "brightness": True,
        "brightness_scale": 100,
        "payload_available": "online",
        "payload_not_available": "offline",
        "device": {
            "identifiers": [DEVICE_ID],
            "name": "Smart Light",
            "model": "ESP32 Light",
            "manufacturer": "MicroPython Virtual Devices",
            "sw_version": "1.0"
        }
    }
    client.publish(TOPIC_DISCOVERY, json.dumps(config), retain=True)

def main():
    connect_wifi()
    
    client = MQTTClient(DEVICE_ID, BROKER, PORT, keepalive=60)
    client.set_callback(on_message)
    
    while True:
        try:
            client.connect()
            print('已连接到MQTT服务器')
            
            # 发布发现配置和可用性状态
            publish_discovery(client)
            client.publish(TOPIC_AVAILABILITY, "online", retain=True)
            
            # 订阅控制主题
            client.subscribe(TOPIC_SET)
            
            # 发布初始状态
            publish_state(client, 0)
            
            while True:
                client.check_msg()
                time.sleep(0.1)
                
        except Exception as e:
            print(f'错误: {e}')
            time.sleep(5)
            continue

if __name__ == "__main__":
    main() 