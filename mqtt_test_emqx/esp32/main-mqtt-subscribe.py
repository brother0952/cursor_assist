from machine import Pin
import network
import time
from umqtt.simple import MQTTClient
import json
import random

# WiFi配置
WIFI_SSID = "HUAWEI-P107NL"
WIFI_PASSWORD = "12871034"

# MQTT配置
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
CLIENT_ID = f"esp32-s3-{random.randint(0, 1000)}"
TOPIC_PUBLISH = b"esp32/sensors"
TOPIC_SUBSCRIBE = b"test/temperature"  # 订阅PC发送的消息

# LED引脚配置
led = Pin(2, Pin.OUT)  # 根据你的ESP32-S3板子调整引脚号

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
        # 如果收到的温度大于30度，点亮LED
        if "temperature" in data and data["temperature"] > 30:
            led.value(1)
        else:
            led.value(0)
    except:
        print('消息格式错误')

def main():
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
                # 发送传感器数据
                sensor_data = {
                    "device_id": CLIENT_ID,
                    "temperature": 25 + random.random() * 10,  # 模拟温度数据
                    "humidity": 60 + random.random() * 20,     # 模拟湿度数据
                }
                client.publish(TOPIC_PUBLISH, json.dumps(sensor_data))
                print(f'已发布消息: {sensor_data}')
                
                # 检查是否有新消息
                client.check_msg()
                
                time.sleep(2)
                
        except Exception as e:
            print(f'错误: {e}')
            time.sleep(5)
            continue

if __name__ == '__main__':
    main() 