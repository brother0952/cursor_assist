
from microdot import Microdot


import machine

import network

sta_if = network.WLAN(network.STA_IF)

print(sta_if.ifconfig()[0])



app = Microdot()

pin = machine.Pin(2, machine.Pin.OUT)

ledoff=1
ledon=0

#pin.value(1) # off



@app.route('/')
def index(request):
    return 'jw Hello, world!'

@app.get('/users/<username>')
def get_user(request, username):
    ''' 192.168.3.55/users/aa '''
    return 'User: ' + username
    

@app.get('/led/<int:id>')
def get_user(request, id):
    if id!=0:
        pin.value(ledon)
    else:
        pin.value(ledoff)
    return 'led: ' + ' (' + str(id) + ')'


app.run(host='0.0.0.0', port=80, debug=False, ssl=None)

