import paho.mqtt.client as mqtt
import time
import json
import random

# MQTT配置
BROKER = "broker.emqx.io"
PORT = 1883
DEVICE_ID = "motion_sensor_01"
DISCOVERY_PREFIX = "homeassistant"

# 主题
TOPIC_STATE = f"home/motion/{DEVICE_ID}/state"
TOPIC_AVAILABILITY = f"home/motion/{DEVICE_ID}/availability"
TOPIC_DISCOVERY = f"{DISCOVERY_PREFIX}/binary_sensor/{DEVICE_ID}/config"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("已连接到MQTT服务器")
        # 发布发现配置
        config = {
            "name": "Motion Sensor",
            "unique_id": DEVICE_ID,
            "device_class": "motion",
            "state_topic": TOPIC_STATE,
            "availability_topic": TOPIC_AVAILABILITY,
            "payload_on": "ON",
            "payload_off": "OFF",
            "device": {
                "identifiers": [DEVICE_ID],
                "name": "Motion Sensor",
                "model": "Virtual Motion Sensor",
                "manufacturer": "Python Virtual Devices"
            }
        }
        client.publish(TOPIC_DISCOVERY, json.dumps(config), retain=True)
        client.publish(TOPIC_AVAILABILITY, "online", retain=True)
    else:
        print(f"连接失败，返回码: {rc}")

def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    
    print(f"正在连接到 {BROKER}...")
    client.connect(BROKER, PORT, 60)
    client.loop_start()

    try:
        while True:
            # 随机模拟运动检测
            motion_detected = random.random() > 0.7
            state = "ON" if motion_detected else "OFF"
            client.publish(TOPIC_STATE, state)
            print(f"运动状态: {state}")
            time.sleep(5)
    except KeyboardInterrupt:
        print("正在停止...")
        client.publish(TOPIC_AVAILABILITY, "offline", retain=True)
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main() 