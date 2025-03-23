import paho.mqtt.client as mqtt
import time
import json
import random

# MQTT配置
BROKER = "broker.emqx.io"
PORT = 1883
DEVICE_ID = "living_room_monitor"
DISCOVERY_PREFIX = "homeassistant"

# Home Assistant MQTT 主题
TOPIC_STATE = f"home/living_room/{DEVICE_ID}/state"
TOPIC_AVAILABILITY = f"home/living_room/{DEVICE_ID}/availability"

# Home Assistant 发现主题
TOPIC_TEMP = f"{DISCOVERY_PREFIX}/sensor/{DEVICE_ID}_temp/config"
TOPIC_HUMID = f"{DISCOVERY_PREFIX}/sensor/{DEVICE_ID}_humid/config"
TOPIC_LIGHT = f"{DISCOVERY_PREFIX}/sensor/{DEVICE_ID}_light/config"

def generate_discovery_configs():
    """生成Home Assistant MQTT发现配置"""
    device_info = {
        "identifiers": [DEVICE_ID],
        "name": "Living Room Monitor",
        "model": "Virtual Monitor v1",
        "manufacturer": "Python Virtual Devices",
        "sw_version": "1.0"
    }
    
    configs = {
        TOPIC_TEMP: {
            "name": "Living Room Temperature",
            "unique_id": f"{DEVICE_ID}_temp",
            "device_class": "temperature",
            "state_class": "measurement",
            "unit_of_measurement": "°C",
            "value_template": "{{ value_json.temperature }}",
            "state_topic": TOPIC_STATE,
            "availability_topic": TOPIC_AVAILABILITY,
            "device": device_info
        },
        TOPIC_HUMID: {
            "name": "Living Room Humidity",
            "unique_id": f"{DEVICE_ID}_humid",
            "device_class": "humidity",
            "state_class": "measurement",
            "unit_of_measurement": "%",
            "value_template": "{{ value_json.humidity }}",
            "state_topic": TOPIC_STATE,
            "availability_topic": TOPIC_AVAILABILITY,
            "device": device_info
        },
        TOPIC_LIGHT: {
            "name": "Living Room Light Level",
            "unique_id": f"{DEVICE_ID}_light",
            "device_class": "illuminance",
            "state_class": "measurement",
            "unit_of_measurement": "lx",
            "value_template": "{{ value_json.light }}",
            "state_topic": TOPIC_STATE,
            "availability_topic": TOPIC_AVAILABILITY,
            "device": device_info
        }
    }
    return configs

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("已连接到MQTT服务器")
        # 发布发现配置
        configs = generate_discovery_configs()
        for topic, config in configs.items():
            client.publish(topic, json.dumps(config), retain=True)
            print(f"已发布配置: {topic}")
        # 发布可用性状态
        client.publish(TOPIC_AVAILABILITY, "online", retain=True)
    else:
        print(f"连接失败，返回码: {rc}")

def publish_state(client):
    """发布传感器状态"""
    state = {
        "temperature": round(21 + random.random() * 4, 1),
        "humidity": round(45 + random.random() * 15, 1),
        "light": round(200 + random.random() * 800, 1),
        "timestamp": time.time()
    }
    client.publish(TOPIC_STATE, json.dumps(state))
    print(f"已发布状态: {state}")

def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    
    print(f"正在连接到 {BROKER}...")
    client.connect(BROKER, PORT, 60)
    client.loop_start()

    try:
        while True:
            publish_state(client)
            time.sleep(10)
    except KeyboardInterrupt:
        print("正在停止...")
        client.publish(TOPIC_AVAILABILITY, "offline", retain=True)
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main() 