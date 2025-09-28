webserver
webrepl


安装 ampy 可以向目标板传文件
pip install adafruit-ampy

ampy --port xx file



wlan.ifconfig()
('192.168.3.55', '255.255.255.0', '192.168.3.1', '192.168.3.1')


import network

wlan = network.WLAN(network.STA_IF) # create station interface
wlan.active(True)       # activate the interface
wlan.scan()             # scan for access points
wlan.isconnected()      # check if the station is connected to an AP
wlan.connect('ssid', 'key') # connect to an AP
wlan.config('mac')      # get the interface's MAC address
wlan.ipconfig('addr4')  # get the interface's IPv4 addresses

ap = network.WLAN(network.AP_IF) # create access-point interface
ap.active(True)         # activate the interface
ap.config(ssid='ESP-AP') # set the SSID of the access point




For online docs please visit http://docs.micropython.org/

For diagnostic information to include in bug reports execute 'import port_diag'.

Basic WiFi configuration:

import network
sta_if = network.WLAN(network.STA_IF); sta_if.active(True)
sta_if.scan()                             # Scan for available access points
sta_if.connect("<AP_name>", "<key>") # Connect to an AP
sta_if.isconnected()                      # Check for successful connection
# Change name/password of ESP8266's AP:
ap_if = network.WLAN(network.AP_IF)
ap_if.config(ssid="<AP_NAME>", security=network.AUTH_WPA_WPA2_PSK, key="<key>")

Control commands:
  CTRL-A        -- on a blank line, enter raw REPL mode
  CTRL-B        -- on a blank line, enter normal REPL mode
  CTRL-C        -- interrupt a running program
  CTRL-D        -- on a blank line, do a soft reset of the board
  CTRL-E        -- on a blank line, enter paste mode

For further help on a specific object, type help(obj)
>>> 



不知原因，webrepl连接后，在web终端不能输入命令



microdot ,在github。放少数文件到设备上就可以引用，创建应用




```python





f = open('data.txt', 'w')
f.write('some data')

f.close()



f = open('data.txt')
f.read()
'some data'
f.close()


import os

os.listdir()

os.mkdir('dir')

os.remove('data.txt')


import network
sta_if = network.WLAN(network.STA_IF)
ap_if = network.WLAN(network.AP_IF)

sta_if.active()
False
ap_if.active()
True

ap_if.ipconfig('addr4')
('192.168.4.1', '255.255.255.0')

sta_if.active(True)


sta_if.connect('<your SSID>', '<your key>')

sta_if.isconnected()

sta_if.ipconfig('addr4')


ap_if.active(False)


def do_connect():
    import network
    sta_if = network.WLAN(network.STA_IF)
    if not sta_if.isconnected():
        print('connecting to network...')
        sta_if.active(True)
        sta_if.connect('<ssid>', '<key>')
        while not sta_if.isconnected():
            pass
    print('network config:', sta_if.ipconfig('addr4'))




# 用socket发http请求

def http_get(url):
    import socket
    _, _, host, path = url.split('/', 3)
    addr = socket.getaddrinfo(host, 80)[0][-1]
    s = socket.socket()
    s.connect(addr)
    s.send(bytes('GET /%s HTTP/1.0\r\nHost: %s\r\n\r\n' % (path, host), 'utf8'))
    while True:
        data = s.recv(100)
        if data:
            print(str(data, 'utf8'), end='')
        else:
            break
    s.close()
# 测试
http_get('http://micropython.org/ks/test.html')



# simple http server


# gpio
pin = machine.Pin(0) # 0 is pin num
pin = machine.Pin(0, machine.Pin.IN, machine.Pin.PULL_UP)

pin.value()

pin = machine.Pin(0, machine.Pin.OUT)
pin.value(0)
pin.value(1)

pin.off()
pin.on()

# 等价



def callback(p):
    print('pin change', p)

from machine import Pin
p0 = Pin(0, Pin.IN)
p2 = Pin(2, Pin.IN)

p0.irq(trigger=Pin.IRQ_FALLING, handler=callback)
p2.irq(trigger=Pin.IRQ_RISING | Pin.IRQ_FALLING, handler=callback)



#On the ESP8266 the pins 0, 2, 4, 5, 12, 13, 14 and 15 all support PWM
import machine
p12 = machine.Pin(12)

pwm12 = machine.PWM(p12)

pwm12.freq(500)
pwm12.duty(512)

pwm12
PWM(12, freq=500, duty=512) # duty max 1023 ,now is 50%

pwm12.deinit()


led = machine.PWM(machine.Pin(2), freq=1000)
import time, math

def pulse(l, t):
    for i in range(20):
        l.duty(int(math.sin(i / 10 * math.pi) * 500 + 500))
        time.sleep_ms(t)

pulse(led, 50) 
for i in range(10):
    pulse(led, 20)


servo = machine.PWM(machine.Pin(12), freq=50)
servo.duty(40)
servo.duty(115)
servo.duty(77)


# ADC
import machine
adc = machine.ADC(0)

adc.read()
58


# power management
import machine
machine.freq()
80000000

machine.freq(160000000)
machine.freq()
160000000

#Deep-sleep mode
import machine

# configure RTC.ALARM0 to be able to wake the device
rtc = machine.RTC()
rtc.irq(trigger=rtc.ALARM0, wake=machine.DEEPSLEEP)

# set RTC.ALARM0 to fire after 10 seconds (waking the device)
rtc.alarm(rtc.ALARM0, 10000)

# put the device to sleep
machine.deepsleep()


if machine.reset_cause() == machine.DEEPSLEEP_RESET:
    print('woke from a deep sleep')
else:
    print('power on or hard reset')






import time

def demo(np):
    n = np.n

    # cycle
    for i in range(4 * n):
        for j in range(n):
            np[j] = (0, 0, 0)
        np[i % n] = (255, 255, 255)
        np.write()
        time.sleep_ms(25)

    # bounce
    for i in range(4 * n):
        for j in range(n):
            np[j] = (0, 0, 128)
        if (i // n) % 2 == 0:
            np[i % n] = (0, 0, 0)
        else:
            np[n - 1 - (i % n)] = (0, 0, 0)
        np.write()
        time.sleep_ms(60)

    # fade in/out
    for i in range(0, 4 * 256, 8):
        for j in range(n):
            if (i // 256) % 2 == 0:
                val = i & 0xff
            else:
                val = 255 - (i & 0xff)
            np[j] = (val, 0, 0)
        np.write()

    # clear
    for i in range(n):
        np[i] = (0, 0, 0)
    np.write()


    



```

# end
