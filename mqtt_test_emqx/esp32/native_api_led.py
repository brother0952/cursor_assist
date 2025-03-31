from microdot import Microdot, Response
from machine import Pin
import neopixel
import network
import json

# WiFi配置
WIFI_SSID = "HUAWEI-P107NL"
WIFI_PASSWORD = "12871034"

# LED配置
LED_PIN = 38
LED_COUNT = 1
np = neopixel.NeoPixel(Pin(LED_PIN), LED_COUNT)

app = Microdot()

def connect_wifi():
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)
        while not wlan.isconnected():
            pass
    return wlan.ifconfig()[0]

@app.route('/api/light', methods=['GET', 'POST'])
def light_control(request):
    if request.method == 'POST':
        try:
            data = request.json
            r = data.get('r', 0)
            g = data.get('g', 0)
            b = data.get('b', 0)
            np[0] = (r, g, b)
            np.write()
            return {'status': 'success'}
        except Exception as e:
            return {'error': str(e)}, 400
    
    # GET请求返回当前状态
    return {'color': {'r': np[0][0], 'g': np[0][1], 'b': np[0][2]}}

if __name__ == '__main__':
    ip = connect_wifi()
    print(f'API server running on http://{ip}:5000')
    app.run(port=5000)
