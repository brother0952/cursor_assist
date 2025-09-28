

from pyb import LED
import time
import gc
from machine import Pin

#led = LED(1) # 1=red, 2=green, 3=yellow, 4=blue
#led_d = LED(2)
#p1=Pin("D5",Pin.Out)
p1=Pin("PB5",Pin.OUT) #red
p2=Pin("PE5",Pin.OUT) # green
#led.toggle()
#led.on()
#led.off()

key = Pin("PA0",Pin.IN,Pin.PULL_DOWN)

while True:
    #led.toggle()
    #led_d.toggle()
    p1.value(not p1.value())
    p2.value(not p2.value())
    time.sleep_ms(200)
    #print(gc.mem_free())
    print(key.value())
    
    
    