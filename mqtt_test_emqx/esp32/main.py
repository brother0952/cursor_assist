from umqtt.simple import MQTTClient
import json
import time
from machine import Pin

# MQTT设置
MQTT_BROKER = "192.168.1.100"  # 替换为你的MQTT服务器IP
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
led = Pin(2, Pin.OUT)  # 使用ESP32板载LED
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

def sub_cb(topic, msg):
    global led_state
    msg = msg.decode()
    print(f"Received: {msg}")
    
    if msg in ["ON", "OFF"]:
        led_state = msg
        if led_state == "ON":
            led.value(1)
            print("LED ON")
        else:
            led.value(0)
            print("LED OFF")
        client.publish(STATE_TOPIC, led_state.encode())

def connect_mqtt():
    global client
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

try:
    connect_mqtt()
    while True:
        client.check_msg()
        time.sleep(0.1)

except Exception as e:
    print(f"Error: {e}")
    machine.reset()
