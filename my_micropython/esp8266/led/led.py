import machine
import time
pin = machine.Pin(2, machine.Pin.OUT)

sleep_wait=1

pin.on()
time.sleep(sleep_wait)
pin.off()
time.sleep(sleep_wait)

#pin.on()
pin.value(1) # off
time.sleep(sleep_wait)
#pin.off()
pin.value(0) # on, test ok
time.sleep(sleep_wait)




pin.value(1) 
time.sleep(sleep_wait)
