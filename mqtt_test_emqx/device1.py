import paho.mqtt.client as mqtt
import time
import json

# MQTT设置
MQTT_BROKER = "localhost"  # 确保这是正确的broker地址
MQTT_PORT = 1883
MQTT_USERNAME = "hass"  # Add your EMQX username here
MQTT_PASSWORD = "hass"  # Add your EMQX password here

# 设备信息
DEVICE_ID = "virtual_device1"
MODEL_NAME = "Virtual Sensor"
MANUFACTURER = "Python Script"

# MQTT主题
DISCOVERY_PREFIX = "homeassistant"
DEVICE_NAME = "Virtual Device 1"
COMPONENT = "sensor"
ENTITY_ID = "status"

# 构建主题
DISCOVERY_TOPIC = f"{DISCOVERY_PREFIX}/{COMPONENT}/{DEVICE_ID}/{ENTITY_ID}/config"
STATE_TOPIC = f"{DISCOVERY_PREFIX}/{COMPONENT}/{DEVICE_ID}/state"
print(f"Discovery topic: {DISCOVERY_TOPIC}")
print(f"State topic: {STATE_TOPIC}")

# 设备发现配置
discovery_message = {
    "name": f"{DEVICE_NAME} Status",
    "unique_id": f"{DEVICE_ID}_{ENTITY_ID}",
    "state_topic": STATE_TOPIC,
    "device": {
        "identifiers": [DEVICE_ID],
        "name": DEVICE_NAME,
        "model": MODEL_NAME,
        "manufacturer": MANUFACTURER,
        "sw_version": "1.0"
    },
    "device_class": "presence",
    "state_class": "measurement",
    "value_template": "{{ value_json.status }}",
    "enabled_by_default": True,
    "payload_available": "online",
    "payload_not_available": "offline"
}

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"Device 1 connected successfully")
        # 发送发现配置
        result = client.publish(DISCOVERY_TOPIC, json.dumps(discovery_message), retain=True, qos=1)
        print(f"Discovery message sent: {discovery_message}")
        print(f"Discovery topic: {DISCOVERY_TOPIC}")
        print(f"Publish result: {result.rc}")
        
        # 发送可用性消息
        availability_topic = f"{DISCOVERY_PREFIX}/{COMPONENT}/{DEVICE_ID}/availability"
        client.publish(availability_topic, "online", retain=True, qos=1)
    else:
        print(f"Failed to connect, return code: {rc}")

def on_message(client, userdata, msg):
    print(f"Device 1 received message from {msg.topic}: {msg.payload.decode()}")

# 使用 MQTT v5 协议
client = mqtt.Client(protocol=mqtt.MQTTv311)
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)  # Set authentication credentials
client.on_connect = on_connect
client.on_message = on_message

def on_disconnect(client, userdata, rc):
    if rc != 0:
        print(f"Unexpected disconnection. Return code: {rc}")
    else:
        print("Successfully disconnected")

client.on_disconnect = on_disconnect

try:
    print(f"Attempting to connect to broker at {MQTT_BROKER}:{MQTT_PORT} with username: {MQTT_USERNAME}")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    # 定期发送状态信息
    while True:
        try:
            state = {
                "status": "online",
                "timestamp": time.time()
            }
            result = client.publish(STATE_TOPIC, json.dumps(state), qos=1)
            if result.rc == 0:
                print(f"Device 1 sent state: {state}")
                print(f"State topic: {STATE_TOPIC}")
            else:
                print(f"Failed to publish state, result code: {result.rc}")
            time.sleep(5)
        except Exception as e:
            print(f"Error in publishing state: {e}")

except Exception as e:
    print(f"Connection error occurred: {e}")
finally:
    print("Cleaning up connection...")
    client.loop_stop()
    client.disconnect()