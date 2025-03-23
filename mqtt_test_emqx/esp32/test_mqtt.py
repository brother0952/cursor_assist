from umqtt.simple import MQTTClient
import time

client = MQTTClient("test", "broker.emqx.io")
client.connect()
while True:
    client.publish(b"test/state", b"123")
    time.sleep(5) 