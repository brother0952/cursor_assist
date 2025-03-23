from machine import Pin
import network
import time
from umqtt.simple import MQTTClient
import json
import random
import neopixel

# WiFi配置
WIFI_SSID = "HUAWEI-P107NL"
WIFI_PASSWORD = "12871034"

# MQTT配置 - EMQX公共服务器
MQTT_BROKER = "broker.emqx.io"  # 改用EMQX公共服务器
MQTT_PORT = 1883
MQTT_USER = None    # EMQX公共服务器不需要认证
MQTT_PASSWORD = None
CLIENT_ID = f"esp32-s3-{random.randint(0, 1000)}"

# MQTT主题 - 添加更明确的主题前缀
DISCOVERY_PREFIX = "homeassistant"
DEVICE_NAME = "esp32_s3_sensor"
DEVICE_ID = CLIENT_ID.replace("-", "_")

# 使用更具体的主题名
BASE_TOPIC = f"esp32/{DEVICE_ID}"
STATE_TOPIC = f"{BASE_TOPIC}/state"
TEMP_CONFIG_TOPIC = f"{DISCOVERY_PREFIX}/sensor/{DEVICE_ID}_temp/config"
HUMID_CONFIG_TOPIC = f"{DISCOVERY_PREFIX}/sensor/{DEVICE_ID}_humid/config"

# WS2812 LED配置
LED_PIN = 38
LED_COUNT = 1
np = neopixel.NeoPixel(Pin(LED_PIN), LED_COUNT)

def set_led_status(status):
    """设置LED状态指示
    连接中: 蓝色
    已连接: 绿色
    错误: 红色
    """
    colors = {
        'connecting': (0, 0, 255),  # 蓝色
        'connected': (0, 255, 0),   # 绿色
        'error': (255, 0, 0)        # 红色
    }
    np[0] = colors.get(status, (0, 0, 0))
    np.write()

def connect_wifi():
    """连接WiFi"""
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print('连接到WiFi...')
        set_led_status('connecting')
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)
        while not wlan.isconnected():
            time.sleep(1)
    print('WiFi已连接')
    print('IP地址:', wlan.ifconfig()[0])
    set_led_status('connected')

def generate_config_payloads():
    """生成Home Assistant MQTT发现配置"""
    device_info = {
        "identifiers": [DEVICE_ID],
        "name": DEVICE_NAME,
        "model": "ESP32-S3",
        "manufacturer": "Espressif",
        "sw_version": "1.0"
    }
    
    base_config = {
        "device": device_info,
        "state_topic": STATE_TOPIC,
        "availability_topic": f"{BASE_TOPIC}/availability",
        "payload_available": "online",
        "payload_not_available": "offline"
    }
    
    temp_config = {
        **base_config,
        "name": f"{DEVICE_NAME} Temperature",
        "unique_id": f"{DEVICE_ID}_temp",
        "unit_of_measurement": "°C",
        "value_template": "{{ value_json.temperature }}",
        "device_class": "temperature",
        "state_class": "measurement"
    }
    
    humid_config = {
        **base_config,
        "name": f"{DEVICE_NAME} Humidity",
        "unique_id": f"{DEVICE_ID}_humid",
        "unit_of_measurement": "%",
        "value_template": "{{ value_json.humidity }}",
        "device_class": "humidity",
        "state_class": "measurement"
    }
    
    return temp_config, humid_config

def publish_discovery_configs(client):
    """发布Home Assistant发现配置"""
    temp_config, humid_config = generate_config_payloads()
    
    # 发布配置
    client.publish(TEMP_CONFIG_TOPIC, json.dumps(temp_config), retain=True)
    client.publish(HUMID_CONFIG_TOPIC, json.dumps(humid_config), retain=True)
    print("已发布Home Assistant发现配置")

def create_mqtt_client():
    """创建MQTT客户端"""
    client = MQTTClient(
        CLIENT_ID, 
        MQTT_BROKER,
        MQTT_PORT,
        keepalive=60
    )
    return client

def connect_mqtt(client):
    """连接到MQTT服务器"""
    try:
        print(f'正在连接到MQTT服务器 {MQTT_BROKER}:{MQTT_PORT}...')
        # 设置超时
        client.connect(clean_session=True, timeout=10.0)
        print('已连接到Home Assistant MQTT服务器')
        return True
    except OSError as e:
        print(f'MQTT连接错误: {e}')
        return False

def publish_state(client, sensor_data):
    """发布状态并打印详细信息"""
    try:
        print("\n发布状态:")
        print(f"主题: {STATE_TOPIC}")
        print(f"数据: {sensor_data}")
        print(f"配置主题: {TEMP_CONFIG_TOPIC}")
        client.publish(STATE_TOPIC, json.dumps(sensor_data))
        return True
    except Exception as e:
        print(f"发布失败: {e}")
        return False

def main():
    # 初始化LED
    set_led_status('connecting')
    
    # 连接WiFi
    connect_wifi()
    
    while True:
        try:
            # 创建新的客户端实例
            client = create_mqtt_client()
            
            # 尝试连接，如果失败则重试
            if not connect_mqtt(client):
                print("连接失败，5秒后重试...")
                set_led_status('error')
                time.sleep(5)
                continue
                
            set_led_status('connected')
            
            # 发布发现配置
            try:
                publish_discovery_configs(client)
            except Exception as e:
                print(f'发布配置失败: {e}')
                client.disconnect()
                continue
            
            # 主循环
            while True:
                try:
                    sensor_data = {
                        "temperature": round(25 + random.random() * 10, 1),
                        "humidity": round(60 + random.random() * 20, 1),
                        "device_id": DEVICE_ID
                    }
                    
                    if not publish_state(client, sensor_data):
                        break
                    
                    time.sleep(30)
                    
                except Exception as e:
                    print(f'发布数据失败: {e}')
                    break  # 跳出内层循环，重新连接
                    
        except Exception as e:
            print(f'MQTT错误: {type(e).__name__} - {str(e)}')
            set_led_status('error')
            print('5秒后重试...')
            time.sleep(5)

if __name__ == '__main__':
    main() 