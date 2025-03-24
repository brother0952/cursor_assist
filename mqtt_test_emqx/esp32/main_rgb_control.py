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
MQTT_BROKER = "192.168.3.42"
MQTT_PORT = 1883
MQTT_USERNAME = "hass"
MQTT_PASSWORD = "hass"
CLIENT_ID = "esp32_rgb_led"

# 设备信息
DEVICE_ID = "esp32_rgb_device"
DEVICE_NAME = "ESP32 RGB LED"

# MQTT主题
DISCOVERY_PREFIX = "homeassistant"
COMPONENT = "light"
ENTITY_ID = "rgb_led"

STATE_TOPIC = f"{DISCOVERY_PREFIX}/{COMPONENT}/{DEVICE_ID}/state"
COMMAND_TOPIC = f"{DISCOVERY_PREFIX}/{COMPONENT}/{DEVICE_ID}/set"
DISCOVERY_TOPIC = f"{DISCOVERY_PREFIX}/{COMPONENT}/{DEVICE_ID}/{ENTITY_ID}/config"

# LED状态变量
led_state = "OFF"
led_brightness = 255
led_color = {"r": 255, "g": 255, "b": 255}

# 设备发现配置
discovery_message = {
    "name": DEVICE_NAME,
    "unique_id": f"{DEVICE_ID}_{ENTITY_ID}",
    "state_topic": STATE_TOPIC,
    "command_topic": COMMAND_TOPIC,
    "schema": "json",
    "brightness": True,
    "rgb": True,
    "device": {
        "identifiers": [DEVICE_ID],
        "name": DEVICE_NAME,
        "model": "ESP32",
        "manufacturer": "Espressif",
        "sw_version": "1.0"
    }
}

# WS2812 LED配置
LED_PIN = 38
LED_COUNT = 1
np = neopixel.NeoPixel(Pin(LED_PIN), LED_COUNT)

def set_led(state, r=255, g=255, b=255, brightness=255):
    if state:
        # 应用亮度
        r = (r * brightness) // 255
        g = (g * brightness) // 255
        b = (b * brightness) // 255
        np[0] = (r, g, b)
    else:
        np[0] = (0, 0, 0)
    np.write()

def sub_cb(topic, msg):
    global led_state, led_brightness, led_color
    try:
        data = json.loads(msg.decode())
        state_changed = False
        
        if "state" in data:
            led_state = data["state"]
            state_changed = True
        if "brightness" in data:
            led_brightness = data["brightness"]
            state_changed = True
        if "color" in data:
            led_color = data["color"]
            state_changed = True
        
        if state_changed:
            if led_state == "ON":
                set_led(True, 
                       led_color["r"], 
                       led_color["g"], 
                       led_color["b"], 
                       led_brightness)
            else:
                set_led(False)
            
            # 发送状态更新
            state_message = {
                "state": led_state,
                "brightness": led_brightness,
                "color": led_color
            }
            client.publish(STATE_TOPIC, json.dumps(state_message).encode())
            
    except Exception as e:
        print(f"Error processing message: {e}")

def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print('连接到WiFi...')
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)
        while not wlan.isconnected():
            time.sleep(1)
    print('WiFi已连接')
    print('IP地址:', wlan.ifconfig()[0])

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
        initial_state = {
            "state": led_state,
            "brightness": led_brightness,
            "color": led_color
        }
        client.publish(STATE_TOPIC, json.dumps(initial_state).encode(), retain=True)
    except Exception as e:
        print(e)

def main():
    try:
        connect_wifi()
        connect_mqtt()
        while True:
            client.check_msg()
            time.sleep(0.1)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
