import paho.mqtt.client as mqtt
import time
import json
import random

from device_config import *

# MQTT设置
MQTT_PORT = 1883

# 设备信息
DEVICE_ID = "virtual_device2"
MODEL_NAME = "Environment Sensor"
MANUFACTURER = "Python Script sensor 1"
DEVICE_NAME = "device name sensor1"
COMPONENT = "sensor"    # ** 组件类型 **

# MQTT主题
DISCOVERY_PREFIX = "homeassistant"

# 传感器配置
SENSORS = {
    "temperature1": {
        "name": "Temperature",
        "device_class": "temperature",
        "state_class": "measurement",
        "unit_of_measurement": "°C",
        "value_template": "{{ value_json.temperature }}"
    },
    "temperature2": {
        "name": "Temperature",
        "device_class": "temperature",
        "state_class": "measurement",
        "unit_of_measurement": "°C",
        "value_template": "{{ value_json.temperature }}"
    },
    "humidity": {
        "name": "Humidity",
        "device_class": "humidity",
        "state_class": "measurement",
        "unit_of_measurement": "%",
        "value_template": "{{ value_json.humidity }}"
    }
}

def create_discovery_message(sensor_type, config):
    return {
        "name": f"{DEVICE_NAME} {config['name']}",
        "unique_id": f"{DEVICE_ID}_{sensor_type}",
        "state_topic": f"{DISCOVERY_PREFIX}/{COMPONENT}/{DEVICE_ID}/state",
        "device_class": config["device_class"],
        "state_class": config["state_class"],
        "unit_of_measurement": config["unit_of_measurement"],
        "value_template": config["value_template"],
        "device": {
            "identifiers": [DEVICE_ID],
            "name": DEVICE_NAME,
            "model": MODEL_NAME,
            "manufacturer": MANUFACTURER,
            "sw_version": "1.0"
        }
    }

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"Device 2 connected successfully")
        # 发送每个传感器的发现配置
        for sensor_type, config in SENSORS.items():
            discovery_topic = f"{DISCOVERY_PREFIX}/{COMPONENT}/{DEVICE_ID}/{sensor_type}/config"
            discovery_message = create_discovery_message(sensor_type, config)
            client.publish(discovery_topic, json.dumps(discovery_message), retain=True, qos=1)
            print(f"Discovery message sent for {sensor_type}")
    else:
        print(f"Failed to connect, return code: {rc}")

def on_disconnect(client, userdata, rc):
    if rc != 0:
        print(f"Unexpected disconnection. Return code: {rc}")
    else:
        print("Successfully disconnected")

# 设置MQTT客户端
client = mqtt.Client(protocol=mqtt.MQTTv311)
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.on_connect = on_connect
client.on_disconnect = on_disconnect

try:
    print(f"Attempting to connect to broker at {MQTT_BROKER}:{MQTT_PORT}")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    while True:
        # 模拟传感器数据
        state = {
            "temperature1": round(random.uniform(20, 25), 1),
            "temperature2": round(random.uniform(20, 25), 1),
            "humidity": round(random.uniform(40, 60), 1),
            "timestamp": time.time()
        }
        
        result = client.publish(f"{DISCOVERY_PREFIX}/{COMPONENT}/{DEVICE_ID}/state", 
                              json.dumps(state), qos=1)
        
        if result.rc == 0:
            print(f"Published state: {state}")
        else:
            print(f"Failed to publish state, result code: {result.rc}")
            
        time.sleep(1)

except Exception as e:
    print(f"Error occurred: {e}")
finally:
    print("Cleaning up connection...")
    client.loop_stop()
    client.disconnect()