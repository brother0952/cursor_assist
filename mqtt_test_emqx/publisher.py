import paho.mqtt.client as mqtt
import time
import json

# MQTT服务器配置
BROKER = "broker.emqx.io"
PORT = 1883
TOPIC = "test/temperature"

# 连接回调
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("成功连接到MQTT服务器")
    else:
        print(f"连接失败，返回码: {rc}")

# 创建MQTT客户端实例
client = mqtt.Client()
client.on_connect = on_connect

# 连接到服务器
print("正在连接到MQTT服务器...")
client.connect(BROKER, PORT, 60)
client.loop_start()

try:
    while True:
        # 创建模拟数据
        message = {
            "temperature": 25 + (time.time() % 10),
            "humidity": 60 + (time.time() % 20),
            "timestamp": time.time()
        }
        
        # 发布消息
        client.publish(TOPIC, json.dumps(message))
        print(f"已发布消息: {message}")
        time.sleep(2)

except KeyboardInterrupt:
    print("正在停止发布者...")
    client.loop_stop()
    client.disconnect() 