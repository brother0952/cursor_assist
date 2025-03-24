from umqtt.simple import MQTTClient
import json
import time
from machine import Pin
import network
import machine
import neopixel


# WiFi配置
WIFI_SSID = "HUAWEI-P107NL"
WIFI_PASSWORD = "12871034"

# MQTT设置
# MQTT_BROKER = "192.168.1.100"  # 替换为你的MQTT服务器IP
#MQTT_BROKER = "127.0.0.1"
MQTT_BROKER = "192.168.3.42"
#MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_USERNAME = "hass"
MQTT_PASSWORD = "hass"
CLIENT_ID = "esp32_led"

# 设备信息
DEVICE_ID = "esp32_device"
DEVICE_NAME = "ESP32 LED"

# MQTT主题
DISCOVERY_PREFIX = "homeassistant"
COMPONENT = "switch"
ENTITY_ID = "led"

STATE_TOPIC = f"{DISCOVERY_PREFIX}/{COMPONENT}/{DEVICE_ID}/state"
COMMAND_TOPIC = f"{DISCOVERY_PREFIX}/{COMPONENT}/{DEVICE_ID}/set"
DISCOVERY_TOPIC = f"{DISCOVERY_PREFIX}/{COMPONENT}/{DEVICE_ID}/{ENTITY_ID}/config"

# 设置LED引脚
# led = Pin(2, Pin.OUT)  # 使用ESP32板载LED
led_state = "OFF"

# 设备发现配置
discovery_message = {
    "name": DEVICE_NAME,
    "unique_id": f"{DEVICE_ID}_{ENTITY_ID}",
    "state_topic": STATE_TOPIC,
    "command_topic": COMMAND_TOPIC,
    "device": {
        "identifiers": [DEVICE_ID],
        "name": DEVICE_NAME,
        "model": "ESP32",
        "manufacturer": "Espressif",
        "sw_version": "1.0"
    },
    "payload_on": "ON",
    "payload_off": "OFF"
}


# WS2812 LED配置
LED_PIN = 38  # WS2812数据引脚
LED_COUNT = 1  # LED数量
np = neopixel.NeoPixel(Pin(LED_PIN), LED_COUNT)

def set_led(on_off):
    if on_off:
        np[0] = (255, 255, 255)  
    else:    
        np[0] = (0, 0, 0)  
    np.write()  # 更新LED显示

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

def sub_cb(topic, msg):
    global led_state
    msg = msg.decode()
    print(f"Received: {msg}")
    
    if msg in ["ON", "OFF"]:
        led_state = msg
        if led_state == "ON":
            # led.value(1)
            set_led(True)
            print("LED ON")
        else:
            # led.value(0)
            set_led(False)
            print("LED OFF")
        client.publish(STATE_TOPIC, led_state.encode())

def connect_mqtt():
    global client
    try:
        client = MQTTClient(CLIENT_ID, MQTT_BROKER,
                           port=MQTT_PORT,
                           user=MQTT_USERNAME,
                           password=MQTT_PASSWORD)
        client.set_callback(sub_cb)
        client.connect()
        print('Connected to MQTT broker')
    
    
        # 发送发现配置
        client.publish(DISCOVERY_TOPIC, json.dumps(discovery_message).encode(), retain=True)
        # 订阅命令主题
        client.subscribe(COMMAND_TOPIC)
        # 发送初始状态
        client.publish(STATE_TOPIC, led_state.encode(), retain=True)
    except Exception as e:
        print(e)

try:
    connect_wifi()
    
    
    connect_mqtt()
    while True:
        client.check_msg()
        time.sleep(0.1)

except Exception as e:
    print(f"Error: {e}")
    # machine.reset()
