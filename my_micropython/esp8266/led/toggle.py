
import machine
import time
pin = machine.Pin(2, machine.Pin.OUT)


def toggle(p):
   p.value(not p.value())

import time
while True:
    toggle(pin)
    time.sleep_ms(500)
    
    