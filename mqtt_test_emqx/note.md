pip install paho-mqtt



一些公共的mqtt服务器
broker.emqx.io
broker.hivemq.com
broker.mqttdashboard.com
broker.mqtt-dashboard.com
mqtt.eclipseprojects.io
mqtt.eclipse.org
mqtt.eclipse.org
mqtt.eclipse.org




esp32传文件方法
pip install esptool
pip install adafruit-ampy

ampy --port COM3 mkdir umqtt
ampy --port COM3 put esp32/main.py
ampy --port COM3 put esp32/boot.py
ampy --port COM3 put esp32/umqtt/simple.py umqtt/simple.py

使用esp32 右边的typec口