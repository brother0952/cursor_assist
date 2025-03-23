import paho.mqtt.client as mqtt
import time
import json
import random

# MQTT配置
BROKER = "broker.emqx.io"
PORT = 1883
DEVICE_ID = "temp_sensor_01"
TOPIC_STATE = f"home/sensors/{DEVICE_ID}/state"
TOPIC_COMMAND = f"home/sensors/{DEVICE_ID}/command"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("已连接到MQTT服务器")
        # 订阅命令主题
        client.subscribe(TOPIC_COMMAND)
        print(f"已订阅主题: {TOPIC_COMMAND}")
    else:
        print(f"连接失败，返回码: {rc}")

def on_message(client, userdata, msg):
    print(f"收到命令: {msg.topic} {msg.payload.decode()}")
    try:
        command = json.loads(msg.payload.decode())
        if command.get("action") == "get_reading":
            publish_state(client)
    except json.JSONDecodeError:
        print("无效的命令格式")

def publish_state(client):
    state = {
        "device_id": DEVICE_ID,
        "temperature": round(20 + random.random() * 10, 1),
        "humidity": round(50 + random.random() * 20, 1),
        "timestamp": time.time()
    }
    client.publish(TOPIC_STATE, json.dumps(state))
    print(f"已发布状态: {state}")

def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"正在连接到 {BROKER}...")
    client.connect(BROKER, PORT, 60)
    client.loop_start()

    try:
        while True:
            publish_state(client)
            time.sleep(5)
    except KeyboardInterrupt:
        print("正在停止...")
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main() 