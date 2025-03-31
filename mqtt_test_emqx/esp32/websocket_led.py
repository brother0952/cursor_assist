import asyncio
import json
from machine import Pin
import neopixel
import network
from microdot_asyncio import Microdot, WebSocket

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

@app.route('/ws')
async def websocket(request):
    ws = await WebSocket.promote(request)
    
    while True:
        try:
            data = await ws.receive_json()
            if 'color' in data:
                color = data['color']
                np[0] = (color['r'], color['g'], color['b'])
                np.write()
                await ws.send_json({'status': 'ok'})
        except:
            break
    
    return ''

@app.route('/')
def home(request):
    return '''
    <html>
    <body>
        <div>
            R: <input type="range" id="r" min="0" max="255" value="0"><br>
            G: <input type="range" id="g" min="0" max="255" value="0"><br>
            B: <input type="range" id="b" min="0" max="255" value="0">
        </div>
        <script>
            const ws = new WebSocket('ws://' + location.host + '/ws');
            const inputs = ['r','g','b'].map(id => document.getElementById(id));
            inputs.forEach(input => {
                input.oninput = () => {
                    ws.send(JSON.stringify({
                        color: {
                            r: inputs[0].value,
                            g: inputs[1].value,
                            b: inputs[2].value
                        }
                    }));
                };
            });
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    ip = connect_wifi()
    print(f'Server running on http://{ip}:5000')
    app.run(port=5000)
