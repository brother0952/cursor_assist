import paho.mqtt.client as mqtt
import json
import time

# MQTT服务器配置
BROKER = "broker.emqx.io"  # 更改为EMQ X的公共服务器
PORT = 1883
TOPICS = [
    "test/temperature",    # PC发送的温度数据
    "esp32/sensors"        # ESP32发送的传感器数据
]

# 连接回调
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("成功连接到MQTT服务器")
        # 连接成功后订阅所有主题
        for topic in TOPICS:
            client.subscribe(topic)
            print(f"已订阅主题: {topic}")
    else:
        print(f"连接失败，返回码: {rc}")

# 断开连接回调
def on_disconnect(client, userdata, rc):
    print("与服务器断开连接")
    if rc != 0:
        print("意外断开连接，尝试重新连接...")
        try_reconnect()

# 消息接收回调
def on_message(client, userdata, msg):
    try:
        # 解析接收到的JSON消息
        payload = json.loads(msg.payload.decode())
        print(f"主题: {msg.topic}")
        if msg.topic == "esp32/sensors":
            print(f"ESP32设备ID: {payload.get('device_id', 'unknown')}")
            print(f"温度: {payload.get('temperature')}°C")
            print(f"湿度: {payload.get('humidity')}%")
        else:
            print(f"收到消息: {payload}")
        print("-" * 50)
    except json.JSONDecodeError:
        print(f"收到非JSON消息: {msg.payload.decode()}")

def try_reconnect():
    while True:
        try:
            client.reconnect()
            break
        except:
            print("重连失败，5秒后重试...")
            time.sleep(5)

# 创建MQTT客户端实例
client = mqtt.Client(protocol=mqtt.MQTTv311)  # 明确指定MQTT协议版本
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect

# 连接到服务器
print("正在连接到MQTT服务器...")
try:
    client.connect(BROKER, PORT, 60)
except Exception as e:
    print(f"连接失败: {e}")
    try_reconnect()

# 开始循环，保持连接
try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\n程序被用户中断")
    client.disconnect()
except Exception as e:
    print(f"发生错误: {e}")
    client.disconnect() 