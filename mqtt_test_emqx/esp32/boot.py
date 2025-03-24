# 这个文件在开机时运行
import gc
gc.collect()

import network
import time


def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print('Connecting to WiFi...')
        wlan.connect('your_ssid', 'your_password')  # 替换为你的WiFi信息
        while not wlan.isconnected():
            time.sleep(1)
    print('WiFi connected!')
    print('Network config:', wlan.ifconfig())

connect_wifi()