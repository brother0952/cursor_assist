from machine import Pin
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
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
CLIENT_ID = f"esp32-s3-{random.randint(0, 1000)}"
TOPIC_SUBSCRIBE = b"test/temperature"  # 订阅PC发送的消息

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
        np[0] = (255, 0, 0)  # 红色 (R,G,B)
    elif temp >= 25:
        np[0] = (0, 255, 0)  # 绿色
    else:
        np[0] = (0, 0, 255)  # 蓝色
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

def mqtt_callback(topic, msg):
    """处理收到的MQTT消息"""
    print(f'收到消息 主题:{topic} 消息:{msg.decode()}')
    try:
        data = json.loads(msg.decode())
        if "temperature" in data:
            temperature = data["temperature"]
            print(f'温度: {temperature}°C')
            set_led_color(temperature)
    except Exception as e:
        print('消息处理错误:', e)

def main():
    # 初始化LED为关闭状态
    np[0] = (0, 0, 0)
    np.write()
    
    # 连接WiFi
    connect_wifi()
    
    # 创建MQTT客户端
    client = MQTTClient(CLIENT_ID, MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.set_callback(mqtt_callback)
    
    while True:
        try:
            # 连接MQTT服务器
            client.connect()
            print('已连接到MQTT服务器')
            
            # 订阅主题
            client.subscribe(TOPIC_SUBSCRIBE)
            print(f'已订阅主题: {TOPIC_SUBSCRIBE}')
            
            # 主循环
            while True:
                # 检查是否有新消息
                client.check_msg()
                time.sleep(0.1)  # 小延时防止占用太多CPU
                
        except Exception as e:
            print(f'错误: {e}')
            print('5秒后重试...')
            # 连接错误时LED显示黄色
            np[0] = (255, 255, 0)
            np.write()
            time.sleep(5)
            continue

if __name__ == '__main__':
    main() 