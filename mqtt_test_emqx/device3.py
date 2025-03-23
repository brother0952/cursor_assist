import paho.mqtt.client as mqtt
import time
import json

# MQTT设置
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_USERNAME = "hass"
MQTT_PASSWORD = "hass"

# 设备信息
DEVICE_ID = "virtual_device3"
MODEL_NAME = "Virtual LED"
MANUFACTURER = "Python Script"
DEVICE_NAME = "Virtual LED Switch"

# MQTT主题
DISCOVERY_PREFIX = "homeassistant"
COMPONENT = "switch"
ENTITY_ID = "led"

# 构建主题
STATE_TOPIC = f"{DISCOVERY_PREFIX}/{COMPONENT}/{DEVICE_ID}/state"
COMMAND_TOPIC = f"{DISCOVERY_PREFIX}/{COMPONENT}/{DEVICE_ID}/set"
DISCOVERY_TOPIC = f"{DISCOVERY_PREFIX}/{COMPONENT}/{DEVICE_ID}/{ENTITY_ID}/config"

# LED状态
led_state = "OFF"

# 设备发现配置
discovery_message = {
    "name": f"{DEVICE_NAME}",
    "unique_id": f"{DEVICE_ID}_{ENTITY_ID}",
    "state_topic": STATE_TOPIC,
    "command_topic": COMMAND_TOPIC,
    "device": {
        "identifiers": [DEVICE_ID],
        "name": DEVICE_NAME,
        "model": MODEL_NAME,
        "manufacturer": MANUFACTURER,
        "sw_version": "1.0"
    },
    "payload_on": "ON",
    "payload_off": "OFF",
    "state_on": "ON",
    "state_off": "OFF",
    "icon": "mdi:led-on"
}

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("Connected successfully")
        # 发送发现配置
        client.publish(DISCOVERY_TOPIC, json.dumps(discovery_message), retain=True, qos=1)
        # 订阅命令主题
        client.subscribe(COMMAND_TOPIC)
        # 发送初始状态
        client.publish(STATE_TOPIC, led_state, retain=True)
    else:
        print(f"Failed to connect, return code: {rc}")

def on_message(client, userdata, msg):
    global led_state
    command = msg.payload.decode()
    print(f"Received command: {command}")
    
    if command in ["ON", "OFF"]:
        led_state = command
        # 发送状态更新
        client.publish(STATE_TOPIC, led_state, retain=True)
        print(f"LED is now {led_state}")
        if led_state == "ON":
            print("💡 LED is illuminated!")
        else:
            print("⚫ LED is turned off!")

# 设置MQTT客户端
client = mqtt.Client(protocol=mqtt.MQTTv311)
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.on_connect = on_connect
client.on_message = on_message

try:
    print(f"Connecting to {MQTT_BROKER}...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()
    
except KeyboardInterrupt:
    print("Stopping...")
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    client.disconnect()
