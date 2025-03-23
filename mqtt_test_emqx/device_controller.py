import paho.mqtt.client as mqtt
import time
import json

# MQTT配置
BROKER = "broker.emqx.io"
PORT = 1883
DEVICE_ID = "controller_01"
SENSOR_ID = "temp_sensor_01"  # 对应传感器的ID

# 主题
TOPIC_SENSOR_STATE = f"home/sensors/{SENSOR_ID}/state"
TOPIC_SENSOR_COMMAND = f"home/sensors/{SENSOR_ID}/command"
TOPIC_CONTROLLER_STATE = f"home/controllers/{DEVICE_ID}/state"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("已连接到MQTT服务器")
        # 订阅传感器状态
        client.subscribe(TOPIC_SENSOR_STATE)
        print(f"已订阅主题: {TOPIC_SENSOR_STATE}")
    else:
        print(f"连接失败，返回码: {rc}")

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        print("\n收到传感器数据:")
        print(f"设备ID: {data['device_id']}")
        print(f"温度: {data['temperature']}°C")
        print(f"湿度: {data['humidity']}%")
        
        # 如果温度超过阈值，发送控制器状态
        if data['temperature'] > 25:
            controller_state = {
                "device_id": DEVICE_ID,
                "action": "cooling",
                "target_temp": 24,
                "timestamp": time.time()
            }
            client.publish(TOPIC_CONTROLLER_STATE, json.dumps(controller_state))
            print(f"\n已发送控制指令: {controller_state}")
    except json.JSONDecodeError:
        print("无效的数据格式")

def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"正在连接到 {BROKER}...")
    client.connect(BROKER, PORT, 60)

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("正在停止...")
        client.disconnect()

if __name__ == "__main__":
    main() 